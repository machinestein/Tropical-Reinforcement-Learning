"""
This is the environment state manager for the LLM agent.
"""
import atexit
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import PIL.Image
import hydra
import math
import random
import numpy as np
import logging

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers
register_resolvers()


def pass_at_k(attempts: int, correct: int, k: int) -> float:
    """Unbiased pass@k estimate: 1 - C(attempts - correct, k) / C(attempts, k)."""
    if not 1 <= k <= attempts:
        raise ValueError(f"pass@{k} requires 1 <= k <= attempts ({attempts})")
    if attempts - correct < k:
        return 1.0
    return 1.0 - math.comb(attempts - correct, k) / math.comb(attempts, k)


def pass_at_k_levels(attempts: int) -> List[int]:
    """Powers of two up to attempts, plus attempts itself."""
    levels = [k for k in (1 << i for i in range(attempts.bit_length())) if k <= attempts]
    if attempts not in levels:
        levels.append(attempts)
    return levels

@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False 
    terminated: bool = False 
    num_actions: int = 0 
    rewards: List[float] = field(default_factory=list) 
    seed: Optional[int] = None 



class EnvStateManager:
    """Manager for the environment state
    The class is responsible for managing multiple (kinds of) environments
    
    """
    def __init__(self, config, mode: str = "train"):
        self.sys_config = config
        self.mode = mode
        self.config = getattr(self.sys_config.es_manager, mode)
        self.env_groups = int(self.config.env_groups)
        self.group_size = self.config.group_size
        seed_cfg = getattr(self.sys_config, "seed", None)
        if seed_cfg is not None:
            self.base_seed = seed_cfg.get(mode, None)
        else:
            self.base_seed = None
        self.seed_counter = 0
        self._init_envs()
        self.rollout_cache = None
        self._executors: Dict[str, ThreadPoolExecutor] = {}
        self._executors_shutdown = False
        self._register_parallel_executors()
        atexit.register(self._shutdown_executors)

    def _init_envs(self):
        """Initialize the environments. train_envs and val_envs are lists of envs:
        Input: tags: ["SimpleSokoban", "HarderSokoban"]; n_groups: [1, 1]; group_size: 16
        Output: envs: List[Dict], each **entry** is a dict with keys: tag, group_id, env_id, env, env_config, status
        Example: [{"tag": "SimpleSokoban", "group_id": 0, "env_id": 0, "env": env, "config": env_config, "status": EnvStatus()},
            ...
            {"tag": "SimpleSokoban", "group_id": 0, "env_id": 15 (group_size - 1), ...},
            {"tag": "HarderSokoban", "group_id": 1, "env_id": 16, ...}
            ...]
        """
        assert sum(self.config.env_configs.n_groups) == self.env_groups, f"Sum of n_groups must equal env_groups. Got sum({self.config.env_configs.n_groups}) != {self.env_groups}"
        assert len(self.config.env_configs.tags) == len(self.config.env_configs.n_groups), f"Number of tags must equal number of n_groups. Got {len(self.config.env_configs.tags)} != {len(self.config.env_configs.n_groups)}"
        self.envs = self._init_env_instances(self.config)

    def _init_env_instances(self, config):
        env_list = []
        done_groups = 0
        for tag, n_group in zip(config.env_configs.tags, config.env_configs.n_groups):
            for env_id in range(done_groups * self.group_size, (done_groups + n_group) * self.group_size):
                cfg_template = self.sys_config.custom_envs[tag]
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                env_kwargs = dict(cfg_template.env_config or {})
                env_kwargs.update(config.get("env_config_overrides", {}).get(tag, {}))
                env_config = REGISTERED_ENV_CONFIGS[env_class](**env_kwargs)
                env_obj = REGISTERED_ENVS[env_class](env_config)
                parallel_friendly = bool(getattr(cfg_template, "parallel_friendly", False))
                max_workers = int(getattr(cfg_template, "max_workers", 1) or 1)
                entry = {'tag': tag, 'group_id': env_id // self.group_size, 'env_id': env_id, 
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj,
                        'parallel_friendly': parallel_friendly, 'max_workers': max_workers}
                env_list.append(entry)
            done_groups += n_group
        return env_list

    def _register_parallel_executors(self):
        tag_seen: Dict[str, dict] = {}
        for entry in self.envs:
            tag = entry["tag"]
            cfg = {
                "parallel_friendly": entry.get("parallel_friendly", False),
                "max_workers": entry.get("max_workers", 1),
            }
            if tag in tag_seen:
                assert tag_seen[tag] == cfg, f"Inconsistent config for tag {tag}: {tag_seen[tag]} vs {cfg}"
            else:
                tag_seen[tag] = cfg

        for tag, cfg in tag_seen.items():
            parallel_friendly = cfg.get('parallel_friendly', False)
            max_workers = cfg.get('max_workers', 1)
            if parallel_friendly and max_workers > 1:
                self._executors[tag] = ThreadPoolExecutor(max_workers=max_workers)

    def reset(self, seed: Optional[int] = None, starts: Optional[List[Dict]] = None):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        """
        def _expand_seed(seed: int):
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)] 
            return sum(seeds, [])

        envs = self.envs
        if starts is not None and len(starts) != len(envs):
            raise ValueError("Explicit rollout starts must match the environment batch size")
        rollout_cache = [{"env_id": entry['env_id'], "history": [], "group_id": entry['group_id'], "tag": entry['tag'], "penalty": 0} for entry in envs]

        if seed is None:
            if self.mode == "train":
                if self.base_seed is not None:
                    seed = self.base_seed + self.seed_counter
                    self.seed_counter += self.env_groups
                else:
                    seed = random.randint(0, 1000000)
            else:
                seed = 123 if self.base_seed is None else self.base_seed
        else:
            if self.mode == "train" and self.base_seed is not None:
                self.seed_counter = seed - self.base_seed + 1
        seeds = _expand_seed(seed)
        if starts is not None:
            seeds = [int(start['seed']) for start in starts]
        elif self.mode == "train" and self.config.get("seed_pool_size") is not None:
            pool_size = int(self.config.seed_pool_size)
            if self.base_seed is None or pool_size < self.env_groups:
                raise ValueError("A recurring seed pool requires a base seed and at least env_groups seeds")
            seeds = [self.base_seed + (s - self.base_seed) % pool_size for s in seeds]

        def _reset_single(entry, single_seed):
            entry['env'].reset(seed=single_seed, mode=self.mode)
            entry['status'] = EnvStatus(seed=single_seed)
            if starts is not None:
                start = starts[entry['env_id']]
                entry['env'].set_state(start['snapshot'])
                history = start['history']
                depth = sum(len(turn.get('actions', [])) for turn in history)
                if depth != entry['env'].num_env_steps or depth >= entry['max_actions_per_traj']:
                    raise ValueError("Restored prefix and remaining action budget are inconsistent")
                if len(history) != depth + 1 or history[-1]['actions_left'] != entry['max_actions_per_traj'] - depth:
                    raise ValueError("Restarts require one action per turn and a complete prefix history")
                if history[-1]['state'] != entry['env'].render():
                    raise ValueError("Restored observation differs from the retained prefix")
                entry['status'].num_actions = depth
                entry['status'].rewards = [t['reward'] for t in history if 'reward' in t]
            return entry['env_id'], self._handle_mm_state(entry['env'].render())

        reset_results = {}
        tag2entries: Dict[str, List[tuple]] = {}
        for single_seed, entry in zip(seeds, envs):
            tag2entries.setdefault(entry['tag'], []).append((entry, single_seed))

        for tag, items in tag2entries.items():
            parallel_friendly = items[0][0].get('parallel_friendly', False)
            max_workers = items[0][0].get('max_workers', 1)
            executor = self._executors.get(tag) if parallel_friendly and max_workers > 1 else None
            if executor is None or len(items) == 1:
                for entry, single_seed in items:
                    env_id, next_state = _reset_single(entry, single_seed)
                    reset_results[env_id] = next_state
            else:
                future_map = {
                    executor.submit(_reset_single, entry, single_seed): entry
                    for entry, single_seed in items
                }
                for future, entry in future_map.items():
                    env_id, next_state = future.result()
                    reset_results[env_id] = next_state

        for cache, env in zip(rollout_cache, envs):
            next_state = reset_results[env['env_id']]
            if starts is None:
                cache['history'] = self._update_cache_history(cache['history'], next_state=next_state, actions_left=env['max_actions_per_traj'], num_actions_info=None)
            else:
                cache['history'] = copy.deepcopy(starts[env['env_id']]['history'])
            
        self.rollout_cache = rollout_cache
        return rollout_cache

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments.
        1. extract valid actions from the action lookup table (if exists) and execute the actions, and update rollout cache
        2. Since rollout does not need to act over done envs, whenever the environment is done, we only update rollout cache, but not output env_outputs.
        Input:
        all_env_inputs: List[Dict]
            {env_id: int, llm_response: str, actions: List[str]}
            NOTE: should use env_id as index for existing some already done envs
        env_outputs: List[Dict]
            {env_id: int, history: List[Dict][{state: str, actions: List[str], reward: float, info: Dict, llm_response: str, llm_raw_response: str, (Optional)images: List[PIL.Image.Image]}]}
        """
        def _execute_actions(env, actions):
            acc_reward, turn_info, turn_done = 0, {}, False
            raw_acc_reward = 0.0
            executed_actions = []
            for action in actions:
                _, reward, done, info = env.step(action)
                acc_reward += reward
                try:
                    raw_acc_reward += float(info.get('raw_reward', 0.0))
                except Exception:
                    pass
                turn_info.update(info) 
                executed_actions.append(action)
                if done:
                    turn_done = True
                    break
            try:
                turn_info['raw_reward'] = float(raw_acc_reward)
            except Exception:
                pass
            return acc_reward, turn_info, turn_done, executed_actions

        def _log_env_state(status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward, turn_done, turn_info, env_input):
            obs = self._handle_mm_state(cur_obs)
            status.num_actions += len(executed_actions)
            status.rewards.append(acc_reward) 
            actions_left = max_actions_per_traj - status.num_actions
            if turn_done:
                status.terminated = True 
                status.truncated = not turn_info.get('success', False)
            history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
                'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
                'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
            })
            return status, history

        envs = self.envs
        env_outputs = []

        def _process_env_input(env_input: Dict) -> Dict:
            acc_reward, turn_info, turn_done = 0, {}, False
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

            valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
            acc_reward, turn_info, turn_done, executed_actions = _execute_actions(env, valid_actions[:actions_left_before])
            no_manager_action = len(valid_actions) == 0
            penalty_delta = 0.0
            if len(valid_actions) != len(env_input['actions']) or not valid_actions:
                penalty_delta = self.sys_config.es_manager.format_penalty
            if no_manager_action:
                turn_info = dict(turn_info)
                turn_info['manager_invalid_action'] = True
                if getattr(self.sys_config.agent_proxy, "terminate_on_invalid_action", False):
                    turn_done = True

            status, history = _log_env_state(entry['status'], self.rollout_cache[env_id]['history'], entry['env'].render(), entry['max_actions_per_traj'], executed_actions, valid_actions, acc_reward, turn_done, turn_info, env_input)
            if no_manager_action and history:
                history[-1]['manager_invalid_action'] = True
            if status.num_actions >= entry['max_actions_per_traj'] and not turn_done:
                status.truncated = True
                status.terminated = True
                turn_done = True

            return {
                'env_id': env_id,
                'status': status,
                'history': history,
                'turn_done': turn_done,
                'penalty_delta': penalty_delta,
            }

        results: List[Optional[Dict]] = [None] * len(all_env_inputs)
        tag2items: Dict[str, List[tuple]] = {}
        for idx, env_input in enumerate(all_env_inputs):
            entry = envs[env_input['env_id']]
            tag2items.setdefault(entry['tag'], []).append((idx, env_input))

        for tag, items in tag2items.items():
            sample_entry = envs[items[0][1]['env_id']]
            parallel_friendly = sample_entry.get('parallel_friendly', False)
            max_workers = sample_entry.get('max_workers', 1)
            executor = self._executors.get(tag) if parallel_friendly and max_workers > 1 else None
            if executor is None or len(items) == 1:
                for idx, env_input in items:
                    results[idx] = _process_env_input(env_input)
            else:
                futures = {executor.submit(_process_env_input, env_input): idx for idx, env_input in items}
                for future, idx in futures.items():
                    results[idx] = future.result()

        for result in results:
            if result is None:
                continue
            env_id = result['env_id']
            entry = envs[env_id]
            if result['penalty_delta']:
                self.rollout_cache[env_id]["penalty"] += result['penalty_delta']
            self.rollout_cache[env_id]['history'] = result['history']
            entry['status'] = result['status']
            if not result['turn_done']:
                env_outputs.append(self.rollout_cache[env_id])

        return env_outputs

    def get_rollout_states(self):
        """Get the final output for all environment"""
        envs = self.envs
        rollout_cache = self.rollout_cache
        TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']

        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            env_metric = {
                'success': float(status.terminated and (not status.truncated)),
                'num_actions': status.num_actions,
            }

            try:
                import numpy as _np
                if hasattr(entry['env'], 'grid'):
                    env_metric['max_tile'] = int(_np.max(entry['env'].grid))
            except Exception:
                pass  

            custom_metric = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    if k == 'success':
                        continue
                    if k not in custom_metric:
                        custom_metric[k] = []
                    try:
                        custom_metric[k].append(float(v))
                    except (ValueError, TypeError):
                        logging.warning(
                            "Skipping non-numeric metric '%s' with value %r for env %s.",
                            k, v, entry['tag']
                        )
            try:
                if 'raw_reward' in custom_metric:
                    env_metric['episodic_return'] = float(np.sum(custom_metric['raw_reward']))
            except Exception:
                pass    

            for k, v in custom_metric.items():
                if "webshop" not in cache['tag'].lower() or ("webshop" in cache['tag'].lower() and k in TURN_LVL_METRICS):
                    env_metric[k] = np.sum(v) / (len(cache['history']) - 1) 
                else:
                    env_metric['traj_sum/' + k] = np.sum(v)
            try:
                if 'score' in custom_metric and len(custom_metric['score']) > 0:
                    env_metric['final_score'] = float(custom_metric['score'][-1])
            except Exception:
                pass

            cache['history'][-1]['metrics'] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache['metrics'] = env_metric
            if entry['tag'] == "MetamathQA":
                cache['correct_answer'] = entry['env'].correct_answer

        group_success = {}
        for entry, cache in zip(envs, rollout_cache):
            key = (entry['tag'], entry['group_id'])
            success_val = cache['metrics'].get(f"{entry['tag']}/success", 0.0)
            group_success.setdefault(key, []).append(success_val)

        group_pass = {}
        for key, succ_list in group_success.items():
            attempts = len(succ_list)
            correct = sum(1 for value in succ_list if value > 0)
            group_pass[key] = {k: pass_at_k(attempts, correct, k) for k in pass_at_k_levels(attempts)}
        for entry, cache in zip(envs, rollout_cache):
            for k, value in group_pass[(entry['tag'], entry['group_id'])].items():
                cache['metrics'][f"{entry['tag']}/pass@{k}"] = value
        return rollout_cache




    def _update_cache_history(self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None: 
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)
        
        entry = {} 
        if isinstance(next_state, str): 
            entry['state'] = next_state
        else: 
            entry['state'] = "<images>" * len(next_state)
            entry['images'] = next_state
        entry['actions_left'] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry['env'].config, 'action_lookup', None)
        if action_lookup is None:
            mapped_actions = actions
        else: 
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
        return mapped_actions
    
    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment
        """
        if isinstance(state, str): 
            return state
        elif isinstance(state, np.ndarray): 
            state = [state]
        results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
        return results
        
    def render(self):
        rendered_list = [entry['env'].render() for entry in self.envs]
        return rendered_list

    def close(self):
        for entry in self.envs:
            entry['env'].close()
        self._shutdown_executors()

    def _shutdown_executors(self):
        if getattr(self, "_executors_shutdown", False):
            return
        self._executors_shutdown = True
        executors = getattr(self, "_executors", None)
        if not executors:
            return
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)

    def __del__(self):
        self._shutdown_executors()




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """
    Unit test for EnvStateManager
    """
    es_manager = EnvStateManager(config, mode="train")
    print("Initializing environments...")
    es_manager.reset(seed=123)

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRunning step for training environments...")
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  
        print(f"Environment {i}:\n{render}\n")

    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go left, go up",
            "llm_response": "Go left, go up",
            "actions": ["left", "up"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go up, go up",
            "llm_response": "Go up, go up",
            "actions": ["up", "up", "up", "up", "up"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRendering final output...")
    final_outputs = es_manager.get_rollout_states()
    print(f"final outputs[:4]: {final_outputs[:4]}")
    
    print("\nClosing environments...")
    es_manager.close()
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
