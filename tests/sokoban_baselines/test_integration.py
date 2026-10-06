"""Real Sokoban/context/driver, scripted generation and a tiny CPU policy."""

import importlib.util
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import Role
from ragen.env.sokoban.env import SokobanEnv
from ragen.env.sokoban.state import SokobanSnapshot
from ragen.llm_agent.agent_proxy import LLMAgentProxy
from ragen.sokoban_baselines.collector import BaselineCollector
from ragen.sokoban_baselines.trainer import RaySokobanBaselineTrainer
from test_algorithms import make_actor

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('baseline_test_tokenizer', ROOT / 'tests/tropic/conftest.py')
tokenizer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokenizer_module)


class ASCIITokenizer(tokenizer_module.CharacterTokenizer):
    def encode(self, text, **kwargs):
        return super().encode(text.encode('ascii', errors='replace').decode(), **kwargs)


class ScriptedProxy(LLMAgentProxy):
    def generate_sequences(self, inputs):
        if inputs.meta_info.get('skip_generation'):
            return inputs
        rows = []
        env_ids = inputs.non_tensor_batch.get('env_ids')
        for i in range(len(inputs)):
            if env_ids is None:  
                text = 'Reconsider the box and push toward its target.</think><answer>Right</answer>'
            else:
                episode = int(env_ids[i])
                manager = self.val_es_manager if inputs.meta_info.get('validate') else self.train_es_manager
                turn = len(manager.rollout_cache[episode]['history']) - 1
                scripts = {0: ['Up', 'Right', 'Left', 'Up', 'Right'], 1: ['Up', 'Down', 'Down', 'Down', 'Down']}
                action = scripts[episode % 2][turn]
                text = f'For path {episode}, choose {action}.</think><answer>{action}</answer>'
            rows.append(self.tokenizer.encode(text + '<|im_end|>'))
        width = max(map(len, rows))
        responses = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(responses)
        for i, row in enumerate(rows):
            responses[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        return DataProto.from_dict(tensors={'responses': responses,
            'attention_mask': torch.cat([inputs.batch['attention_mask'], mask], -1)},
            non_tensors=inputs.non_tensor_batch, meta_info=inputs.meta_info)


class CPUWorker:
    world_size = 1

    def __init__(self, method):
        self.actor = make_actor(torch.nn.Embedding(256, 256), method)
        self.updates = 0

    def compute_log_prob(self, batch):
        with torch.no_grad():
            _, logp = self.actor._forward_micro_batch(batch.batch, 1)
        return DataProto.from_dict(tensors={'old_log_probs': logp})

    def update_actor(self, batch):
        self.updates += 1
        return DataProto(meta_info={'metrics': self.actor.update_policy(batch)})

    def save_checkpoint(self, local_path, *args, **kwargs):
        Path(local_path).mkdir(parents=True)
        torch.save(self.actor.actor_module.state_dict(), Path(local_path) / 'test.pt')


@pytest.fixture(params=['maxrl', 'tstar'])
def setup(request, tmp_path, monkeypatch):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with initialize_config_dir(config_dir=str(ROOT / 'config'), version_base=None):
        cfg = compose(config_name=f'_2_sokoban_{request.param}')
    for mode in ('train', 'val'):
        cfg.es_manager[mode].env_groups = 1
        cfg.es_manager[mode].group_size = 2
        cfg.es_manager[mode].env_configs.n_groups = [1]
    cfg.actor_rollout_ref.rollout.max_model_len = 12000  
    cfg.custom_envs.CoordSokoban.env_config.dim_x = 5
    cfg.custom_envs.CoordSokoban.env_config.dim_y = 5
    cfg.trainer.total_training_steps = 2
    cfg.trainer.test_freq = 1
    cfg.trainer.save_freq = 2
    cfg.trainer.generations_to_log_to_wandb.val = 0
    cfg.trainer.default_local_dir = str(tmp_path / 'checkpoints')
    cfg.trainer.logger = ['console', 'file']
    monkeypatch.setenv('VERL_FILE_LOGGER_PATH', str(tmp_path / 'metrics.jsonl'))

    def reset(env, seed=None, mode=None):
        fixed = np.ones((5, 5), dtype=int)
        fixed[[0, -1], :] = 0
        fixed[:, [0, -1]] = 0
        fixed[1, 3] = 2
        state = fixed.copy()
        state[3, 1], state[1, 2] = 5, 4
        return env.set_state(SokobanSnapshot(fixed.tolist(), state.tolist(), [3, 1],
                            [[[1, 3], [1, 2]]], 0, 0, 0., None, None, env.max_steps, 1))
    monkeypatch.setattr(SokobanEnv, 'reset', reset)
    proxy = ScriptedProxy(cfg, None, ASCIITokenizer())
    worker = CPUWorker(request.param)
    try:
        yield cfg, proxy, worker
    finally:
        proxy.train_es_manager.close()
        proxy.val_es_manager.close()
        torch.set_num_threads(old_threads)


def test_collector_uses_terminal_success_and_preserves_prefilled_thoughts(setup):
    cfg, proxy, worker = setup
    collector = BaselineCollector(cfg, proxy, worker, proxy.tokenizer)
    rows, advantages, pairs, metrics = collector.collect()
    assert metrics[cfg.trainer.method + '/success_rate'] == .5
    assert len(rows) == 10
    assert all(row.reward == (1 if row.episode == 0 else 0) for row in rows)
    assert all(proxy.tokenizer.decode(row.prompt_ids).endswith('<think>') for row in rows)
    assert any(advantages > 0) and any(advantages < 0)
    if cfg.trainer.method == 'maxrl':
        assert not pairs
        np.testing.assert_allclose(advantages, [1, -1] * 5, atol=1e-7)
    else:
        assert metrics['tstar/kl_comparisons'] > 0
        assert metrics['tstar/merged_decisions'] > 0
        assert metrics['tstar/new_grafts'] > 0
        assert metrics['tstar/extra_generated_tokens'] > 0
        for chosen, rejected in pairs:
            assert chosen.prompt_ids == rejected.prompt_ids
            assert any(row.prompt_ids == rejected.prompt_ids for row in rows)
            for row in (chosen, rejected):
                text = proxy.tokenizer.decode(row.completion_ids)
                assert text.endswith('</think>') and '<answer>' not in text
                assert '<think>' not in text  


def test_driver_updates_validates_saves_and_logs_on_real_environment(setup, tmp_path):
    from train import DummyRewardManager
    cfg, proxy, worker = setup
    reward = DummyRewardManager(proxy.tokenizer, num_examine=0)
    trainer = RaySokobanBaselineTrainer(cfg, proxy.tokenizer, {Role.ActorRollout: CPUWorker}, None,
                                      reward_fn=reward, val_reward_fn=reward)
    trainer.agent_proxy, trainer.actor_rollout_wg = proxy, worker
    trainer.collector = BaselineCollector(cfg, proxy, worker, proxy.tokenizer)
    before = worker.actor.actor_module.weight.detach().clone()
    trainer.fit()
    assert worker.updates == 2
    assert not torch.equal(before, worker.actor.actor_module.weight)
    records = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert [r['step'] for r in records] == [0, 1, 2]
    assert any('pass@2' in key for key in records[-1]['data'])
    assert records[-1]['data']['actor/optimizer_steps'] == 1
    assert records[-1]['data']['timing_s/elapsed'] > 0
    assert (tmp_path / 'checkpoints/global_step_2/actor/test.pt').exists()
    assert (tmp_path / 'checkpoints/latest_checkpointed_iteration.txt').read_text() == '2'
    if cfg.trainer.method == 'tstar':
        assert len(trainer.collector.graft_buffer) >= 2
        assert (tmp_path / 'checkpoints/tstar_events.jsonl').read_text()
