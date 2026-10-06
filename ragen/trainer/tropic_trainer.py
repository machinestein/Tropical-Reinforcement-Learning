"""Driver-side TROPIC policy iteration; validation remains ordinary root sampling."""

import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
from omegaconf import OmegaConf
import torch
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import reduce_metrics
from ragen.trainer.tracking import Tracking
from ragen.tropic.batch import basis_rows, make_batch
from ragen.tropic.collector import TropicCollector
from .agent_trainer import RayAgentTrainer


class RayTropicTrainer(RayAgentTrainer):
    def init_agent_proxy(self):
        super().init_agent_proxy()
        tag = self.config.es_manager.train.env_configs.tags[0]
        collector_class = TropicCollector
        if self.config.custom_envs[tag].env_type == 'lean_tropic':
            from ragen.tropic.lean_collector import LeanTropicCollector
            collector_class = LeanTropicCollector
        elif self.config.custom_envs[tag].env_type == 'webshop_tropic':
            from ragen.tropic.webshop_collector import WebShopTropicCollector
            collector_class = WebShopTropicCollector
        elif self.config.custom_envs[tag].env_type == 'sudoku_tropic':
            from ragen.tropic.sudoku_collector import SudokuTropicCollector
            collector_class = SudokuTropicCollector
        elif self.config.custom_envs[tag].env_type == 'countdown_step':
            from ragen.tropic.countdown_collector import CountdownTropicCollector
            collector_class = CountdownTropicCollector
        elif self.config.custom_envs[tag].env_type == 'frozen_lake_tropic':
            from ragen.tropic.frozen_lake_collector import FrozenLakeTropicCollector
            collector_class = FrozenLakeTropicCollector
        self.collector = collector_class(self.config, self.agent_proxy, self._score_edges)

    def _score_edges(self, rows):
        scores = []
        size = int(self.config.tropic.rescore_batch_size)
        for start in range(0, len(rows), size):
            chunk = rows[start:start + size]
            batch = make_batch(chunk, self.tokenizer.pad_token_id, divisor=self.actor_rollout_wg.world_size)
            output = self.actor_rollout_wg.compute_log_prob(batch)
            values = output.batch['old_log_probs'].float().cpu().masked_fill(
                ~batch.batch['response_mask'].bool(), 0.0).sum(-1)
            scores.extend(values[:len(chunk)].tolist())
        return scores

    def _signature(self):
        cfg = self.config
        fields = {
            'tropic': OmegaConf.to_container(cfg.tropic, resolve=True),
            'agent_proxy': OmegaConf.to_container(cfg.agent_proxy, resolve=True),
            'train': OmegaConf.to_container(cfg.es_manager.train, resolve=True),
            'envs': {tag: OmegaConf.to_container(cfg.custom_envs[tag], resolve=True)
                     for tag in cfg.es_manager.train.env_configs.tags},
            'seed': int(cfg.seed.train), 'model': str(cfg.model_path),
            'tokenizer': getattr(self.tokenizer, 'name_or_path', ''),
            'eos': self.tokenizer.eos_token_id, 'pad': self.tokenizer.pad_token_id,
            'context': OmegaConf.to_container(cfg.ctx_manager, resolve=True),
            'model_config': OmegaConf.to_container(cfg.actor_rollout_ref.model, resolve=True),
            'generation': {key: cfg.actor_rollout_ref.rollout[key] for key in
                           ('temperature', 'top_p', 'top_k', 'response_length', 'max_model_len')},
        }
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()

    def _save_checkpoint(self):
        folder = Path(self.config.trainer.default_local_dir) / f'global_step_{self.global_steps}'
        folder.mkdir(parents=True, exist_ok=True)
        state = {
            'signature': self._signature(), 'collector': self.collector.state_dict(),
            'step': self.global_steps, 'python_rng': random.getstate(),
            'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
            'collapse_ema': self.collapse_detector._ema_marginal_std,
            'collapse_ema_seq': self.collapse_detector._ema_marginal_std_seq,
        }
        temporary = folder / 'tropic.pt.tmp'
        torch.save(state, temporary)
        os.replace(temporary, folder / 'tropic.pt')
        super()._save_checkpoint()

    def _load_checkpoint(self):
        mode = self.config.trainer.resume_mode
        folder = None
        if mode == 'auto':
            folder = find_latest_ckpt_path(str(Path(self.config.trainer.default_local_dir).resolve()))
        elif mode == 'resume_path':
            folder = self.config.trainer.resume_from_path
        if folder is not None:
            state = torch.load(Path(folder) / 'tropic.pt', map_location='cpu', weights_only=False)
            if state['signature'] != self._signature():
                raise ValueError("TROPIC checkpoint task, tokenization or collection settings differ")
        super()._load_checkpoint()
        if folder is not None:
            if state['step'] != self.global_steps:
                raise ValueError("Actor and TROPIC memory checkpoint steps differ")
            self.collector.load_state_dict(state['collector'])
            random.setstate(state['python_rng'])
            np.random.set_state(state['numpy_rng'])
            torch.set_rng_state(state['torch_rng'])
            self.collapse_detector._ema_marginal_std = state['collapse_ema']
            self.collapse_detector._ema_marginal_std_seq = state['collapse_ema_seq']

    def fit(self):
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))
        self.global_steps = 0
        output_dir = Path(self.config.trainer.default_local_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._load_checkpoint()
            if self.config.trainer.val_before_train:
                logger.log(self._validate(), step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return
            for step in range(self.global_steps + 1, self.total_training_steps + 1):
                self.global_steps = step
                start = time.perf_counter()
                bases, metrics = self.collector.collect(step)
                if self.collector.root_diagnostics_batch is not None:
                    metrics.update(self.collapse_detector.compute_collapse_metrics(
                        self.collector.root_diagnostics_batch, self.actor_rollout_wg.compute_log_prob, step))
                rows, weights = basis_rows(self.collector.graphs, bases)
                metrics['tropic/update_skipped'] = int(not rows or self.config.tropic.collection_only)
                if rows and not self.config.tropic.collection_only:
                    world = self.actor_rollout_wg.world_size
                    micro = self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
                    batch = make_batch(rows, self.tokenizer.pad_token_id, weights, divisor=world * micro)
                    batch.meta_info['tropic_dp_size'] = world
                    output = self.actor_rollout_wg.update_actor(batch)
                    metrics.update(reduce_metrics(output.meta_info['metrics']))
                    epochs = self.config.actor_rollout_ref.actor.ppo_epochs
                    metrics['tropic/training_completion_tokens'] = sum(len(r.completion_ids) for r in rows) * epochs
                    metrics['tropic/training_input_tokens_including_padding_rows'] = sum(batch.meta_info['global_token_num']) * epochs
                with (output_dir / 'tropic_events.jsonl').open('a') as stream:
                    for event in self.collector.events:
                        stream.write(json.dumps({'step': step, **event}) + '\n')
                freq = self.config.trainer.test_freq
                if freq > 0 and (step % freq == 0 or step == self.total_training_steps):
                    metrics.update(self._validate())
                metrics['tropic/iteration_seconds'] = time.perf_counter() - start
                logger.log(metrics, step=step)
                save_freq = self.config.trainer.save_freq
                if save_freq > 0 and (step % save_freq == 0 or step == self.total_training_steps):
                    self._save_checkpoint()
        finally:
            self.collector.close()
            self.agent_proxy.train_es_manager.close()
            self.agent_proxy.val_es_manager.close()
            if hasattr(logger, 'finish'):
                logger.finish()
