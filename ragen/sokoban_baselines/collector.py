"""Fresh Sokoban trajectories and T-STAR's additional policy queries."""

from collections import defaultdict
from dataclasses import replace

import numpy as np
import torch
from verl import DataProto

from .batch import token_batch
from .core import Decision, build_tree, maxrl_advantages, thought_prefix


class BaselineCollector:
    def __init__(self, config, proxy, worker, tokenizer):
        self.config, self.proxy, self.worker, self.tokenizer = config, proxy, worker, tokenizer
        self.graft_buffer = []
        self.rng = np.random.default_rng(config.seed.train)

    def _record(self, inputs, outputs, env_inputs, before, manager):
        for i, item in enumerate(env_inputs):
            episode = int(item['env_id'])
            entry = manager.envs[episode]
            history = manager.rollout_cache[episode]['history']
            prompt = tuple(inputs.batch['input_ids'][i][inputs.batch['attention_mask'][i].bool()].tolist())
            response = outputs.batch['responses'][i]
            mask = outputs.batch['attention_mask'][i, -len(response):].bool()
            completion = tuple(response[mask].tolist())
            if not completion:
                raise ValueError("Empty sampled response cannot supply a policy gradient")
            path = self.paths[episode]
            if path:
                self.rows[path[-1]].next_prompt_ids = prompt
            actions = tuple(int(a) for turn in history[:-1] for a in turn.get('actions', []))
            self.rows.append(Decision(prompt, completion, episode, int(entry['group_id']), len(path),
                                      actions, observation=str(history[-1]['state'])))
            path.append(len(self.rows) - 1)

    def _generate(self, prompts):
        """One independent extra generation pass; same untruncated sampling policy."""
        if not prompts:
            return []
        width = max(map(len, prompts))
        ids = torch.full((len(prompts), width), self.tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(ids)
        for i, prompt in enumerate(prompts):
            ids[i, -len(prompt):] = torch.tensor(prompt)
            attention[i, -len(prompt):] = 1
        inputs = DataProto.from_dict(tensors={
            'input_ids': ids, 'attention_mask': attention,
            'position_ids': (attention.cumsum(-1) - 1).clamp(min=0),
        }, meta_info={'mode': 'singleturn', 'do_sample': True, 'validate': False,
                      'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id})
        outputs = self.proxy.generate_sequences(inputs)
        completions = []
        for i, response in enumerate(outputs.batch['responses']):
            mask = outputs.batch['attention_mask'][i, -len(response):].bool()
            completions.append(tuple(response[mask].tolist()))
        self.metrics['extra_generated_tokens'] += sum(map(len, completions))
        self.metrics['extra_generation_prompt_tokens'] += sum(map(len, prompts))
        return completions

    def _score(self, rows):
        values = []
        size, world = self.config.sokoban_baseline.score_batch_size, self.worker.world_size
        for start in range(0, len(rows), size):
            chunk = rows[start:start + size]
            padded = chunk + [chunk[0]] * ((-len(chunk)) % world)
            batch = token_batch(padded, self.tokenizer.pad_token_id)
            output = self.worker.compute_log_prob(batch)
            scores = output.batch['old_log_probs'].float().cpu().masked_fill(
                ~batch.batch['response_mask'].bool(), 0.0).sum(-1)
            values.extend(scores[:len(chunk)].tolist())
            self.metrics['extra_scored_tokens'] += sum(len(r.prompt_ids) + len(r.completion_ids) for r in padded)
        return np.asarray(values)

    def _kl(self, left, right):
        source = left.next_prompt_ids
        if source not in self.kl_cache:
            completions = self._generate([source] * self.config.tstar.kl_samples)
            if any(not c for c in completions):
                raise ValueError("Empty KL sample")
            own = self._score([Decision(source, c) for c in completions])
            self.kl_cache[source] = completions, own
        completions, own = self.kl_cache[source]
        other = self._score([Decision(right.next_prompt_ids, c) for c in completions])
        value = float((own - other).mean())
        self.metrics['kl_comparisons'] += 1
        self.metrics['negative_kl_estimates'] += int(value < 0)
        return value

    def _graft(self, tree):
        candidates = []
        token_budget = (self.config.actor_rollout_ref.rollout.max_model_len -
                        self.config.actor_rollout_ref.rollout.response_length)
        for positive, negative, spread in tree.divergences:
            good, bad = self.rows[positive], self.rows[negative]
            rejected = thought_prefix(self.tokenizer, bad.completion_ids, opening_in_prompt=True)
            if not rejected:
                self.metrics['grafts_missing_thought'] += 1
                continue
            context = self.tokenizer.decode(bad.prompt_ids, skip_special_tokens=False)
            best = self.tokenizer.decode(good.completion_ids, skip_special_tokens=True)
            worst = self.tokenizer.decode(bad.completion_ids, skip_special_tokens=True)
            instruction = (
                'Correct the reasoning at the next decision in this Sokoban interaction. '
                'Use the higher-value branch to identify the mistake in the lower-value branch. '
                'Continue the prefilled <think> block with corrected reasoning for the ORIGINAL CONTEXT, '
                'ending with </think>; '
                'do not mention these comparisons, invent future observations, or output an <answer>.\n'
                f'ORIGINAL CONTEXT:\n{context}\nHIGHER-VALUE BRANCH:\n{best}\n'
                f'Its observation: {good.observation}\nLOWER-VALUE BRANCH:\n{worst}\n'
                f'Its observation: {bad.observation}')
            prompt = tuple(self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': instruction}], tokenize=True, add_generation_prompt=True))
            prompt += tuple(self.tokenizer.encode('<think>', add_special_tokens=False))
            if len(prompt) > token_budget:
                self.metrics['grafts_over_context_limit'] += 1
                continue
            candidates.append((prompt, bad, rejected, spread))
        size = self.config.sokoban_baseline.score_batch_size
        for start in range(0, len(candidates), size):
            chunk = candidates[start:start + size]
            responses = self._generate([item[0] for item in chunk])
            for (_, bad, rejected, spread), response in zip(chunk, responses):
                chosen = thought_prefix(self.tokenizer, response, opening_in_prompt=True)
                if not chosen or chosen == rejected:
                    self.metrics['grafts_missing_or_identical_correction'] += 1
                    continue
                pair = (replace(bad, completion_ids=chosen), replace(bad, completion_ids=rejected))
                self.graft_buffer.append(pair)
                self.events.append({'group': bad.group, 'episode': bad.episode, 'depth': bad.depth,
                                    'value_spread': spread, 'prompt_ids': list(bad.prompt_ids),
                                    'chosen_ids': list(chosen), 'rejected_ids': list(rejected)})

    def collect(self):
        self.rows, self.paths = [], defaultdict(list)
        self.metrics, self.events, self.kl_cache = defaultdict(float), [], {}
        meta = {'do_sample': True, 'validate': False, 'compute_collapse': False,
                'eos_token_id': self.tokenizer.eos_token_id, 'pad_token_id': self.tokenizer.pad_token_id}
        rollouts = self.proxy.rollout(DataProto(meta_info=meta), transition_recorder=self._record)
        rewards, groups = [], []
        episode_adv = {}
        for episode, path in self.paths.items():
            entry = self.proxy.train_es_manager.envs[episode]
            status = entry['status']
            reward = float(status.terminated and not status.truncated)
            for i in path:
                self.rows[i].reward = reward
            rewards.append(reward)
            groups.append(int(entry['group_id']))
        metrics = {'train-env/' + k: v for k, v in rollouts.meta_info['metrics'].items()}
        self.metrics.update(success_rate=float(np.mean(rewards)), trajectories=len(rewards),
                            decisions=len(self.rows), generated_tokens=sum(len(r.completion_ids) for r in self.rows))
        if self.config.trainer.method == 'maxrl':
            values = maxrl_advantages(rewards, groups, self.config.maxrl.epsilon)
            episode_adv = dict(zip(self.paths, values))
            advantages = np.asarray([episode_adv[row.episode] for row in self.rows])
            self.metrics['effective_truncation'] = self.config.es_manager.train.group_size - 1
            pairs = []
        else:
            tree = build_tree(self.rows, self._kl, self.config.tstar.kl_threshold,
                              self.config.tstar.gamma, self.config.tstar.divergence_threshold)
            advantages = tree.advantages
            self.metrics.update(tree_nodes=len(tree.members), merged_decisions=len(self.rows) - len(tree.members),
                                divergences=len(tree.divergences))
            self._graft(tree)
            self.metrics.update(new_grafts=len(self.events), graft_buffer_size=len(self.graft_buffer))
            limit = self.config.tstar.graft_batch_size
            if limit and len(self.graft_buffer) > limit:
                selected = self.rng.choice(len(self.graft_buffer), limit, replace=False)
                pairs = [self.graft_buffer[i] for i in selected]
            else:
                pairs = list(self.graft_buffer)
            self.metrics['training_graft_pairs'] = len(pairs)
        self.metrics['nonzero_advantage_fraction'] = float(np.mean(advantages != 0))
        prefix = self.config.trainer.method + '/'
        metrics.update({prefix + k: v for k, v in self.metrics.items()})
        return self.rows, advantages, pairs, metrics
