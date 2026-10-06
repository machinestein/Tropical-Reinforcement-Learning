import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.ray_trainer import Role
from ragen.tropic.batch import verified_path_loss
from ragen.tropic.collector import TropicCollector
from ragen.trainer.collapse_metrics import CollapseDetector
from ragen.trainer.tropic_trainer import RayTropicTrainer
from test_collector import proxy  


class CPUWorker:
    """Exercise driver data flow without Ray, model downloads or GPU kernels."""

    world_size = 1

    def __init__(self):
        self.value = torch.nn.Parameter(torch.tensor(-.5))
        self.optimizer = torch.optim.SGD([self.value], lr=.01)
        self.updates = 0

    def compute_log_prob(self, batch):
        values = self.value.detach().expand_as(batch.batch['responses']).clone()
        return DataProto.from_dict(tensors={'old_log_probs': values})

    def update_actor(self, batch):
        self.optimizer.zero_grad()
        loss = verified_path_loss(self.value.expand_as(batch.batch['responses']),
                                  batch.batch['response_mask'], batch.batch['tropic_weights'])
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        return DataProto(meta_info={'metrics': {'actor/verified_nll': [float(loss.detach())]}})

    def save_checkpoint(self, local_path, remote_path, step, **kwargs):
        Path(local_path).mkdir(parents=True, exist_ok=True)
        torch.save(self.value.detach(), Path(local_path) / 'test_weight.pt')

    def load_checkpoint(self, local_path, **kwargs):
        if local_path is not None:
            with torch.no_grad():
                self.value.copy_(torch.load(Path(local_path) / 'test_weight.pt', weights_only=True))


@pytest.fixture
def trainer(config, proxy, tmp_path):
    config.trainer.default_local_dir = str(tmp_path)
    config.trainer.val_before_train = False
    config.trainer.test_freq = -1
    config.trainer.save_freq = 1
    instance = RayTropicTrainer(config, proxy.tokenizer, {Role.ActorRollout: CPUWorker}, None)
    instance.total_training_steps = 1
    instance.global_steps = 0
    assert not instance.use_critic
    assert not instance.use_reference_policy
    instance.tokenizer = proxy.tokenizer
    instance.agent_proxy = proxy
    instance.actor_rollout_wg = CPUWorker()
    instance.collector = TropicCollector(config, proxy, instance._score_edges)
    instance.collapse_detector = CollapseDetector(first_turn_enabled=True, context_window_mode='single_turn')
    yield instance
    instance.collector.close()


def test_training_driver_updates_and_resumes_actor_and_memory(trainer):
    initial = float(trainer.actor_rollout_wg.value.detach())
    trainer.fit()
    assert trainer.actor_rollout_wg.updates == 1
    assert float(trainer.actor_rollout_wg.value.detach()) > initial
    folder = Path(trainer.config.trainer.default_local_dir)
    assert (folder / 'global_step_1' / 'tropic.pt').exists()
    assert (folder / 'latest_checkpointed_iteration.txt').read_text() == '1'
    events = [json.loads(line) for line in (folder / 'tropic_events.jsonl').read_text().splitlines()]
    assert {event['origin'] for event in events} == {'root', 'composition'}
    memory = trainer.collector.state_dict()
    expected_rng = (random.random(), np.random.random(), torch.rand(1))
    trainer.collector.graphs.clear()
    trainer.collector.cursor = 0
    trainer.config.trainer.resume_mode = 'auto'
    trainer.global_steps = 0
    trainer._load_checkpoint()
    assert trainer.global_steps == 1
    assert trainer.collector.state_dict() == memory
    assert random.random() == expected_rng[0]
    assert np.random.random() == expected_rng[1]
    torch.testing.assert_close(torch.rand(1), expected_rng[2])


def test_training_driver_writes_numeric_json_metrics(trainer, tmp_path, monkeypatch):
    path = tmp_path / 'metrics.jsonl'
    monkeypatch.setenv('VERL_FILE_LOGGER_PATH', str(path))
    trainer.config.trainer.logger = ['console', 'file']

    trainer.fit()

    record = json.loads(path.read_text())
    assert record['step'] == 1
    assert isinstance(record['data']['actor/verified_nll'], float)
    assert record['data']['tropic/update_skipped'] == 0
    assert trainer.actor_rollout_wg.updates == 1


@pytest.mark.parametrize('collection_only,no_success', [(True, False), (False, True)])
def test_driver_skips_updates_for_pilot_or_no_verified_solutions(trainer, collection_only, no_success):
    trainer.config.tropic.collection_only = collection_only
    if no_success:
        trainer.agent_proxy.scripts = {0: ['invalid'], 1: ['invalid']}
    trainer.fit()
    assert trainer.actor_rollout_wg.updates == 0


def test_resume_rejects_changed_interface_before_loading_actor(trainer):
    trainer.collector.collect(1)
    trainer.global_steps = 1
    trainer._save_checkpoint()
    trainer.config.trainer.resume_mode = 'auto'
    trainer.config.agent_proxy.max_turn += 1
    with pytest.raises(ValueError, match='settings differ'):
        trainer._load_checkpoint()


def test_first_turn_mi_can_run_on_explicit_root_samples_in_state_mode(trainer):
    trainer.collector.collect(1)
    batch = trainer.collector.root_diagnostics_batch
    batch.non_tensor_batch['group_ids'] = np.array([0, 1])
    result = trainer.collapse_detector.compute_collapse_metrics(batch, trainer.actor_rollout_wg.compute_log_prob, 1)
    assert result['collapse/first_turn_num_valid'] == 2
    assert any(key.startswith('collapse_first_turn_sample/') for key in result)
    batch.meta_info.pop('root_only_diagnostics')
    assert trainer.collapse_detector.compute_collapse_metrics(batch, trainer.actor_rollout_wg.compute_log_prob, 1) == {}
