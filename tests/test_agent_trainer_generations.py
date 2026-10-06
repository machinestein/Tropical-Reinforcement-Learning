import json
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from verl import DataProto
from ragen.trainer.agent_trainer import RayAgentTrainer
from ragen.trainer.tropic_trainer import RayTropicTrainer


@pytest.fixture(params=[RayAgentTrainer, RayTropicTrainer], ids=['ppo', 'tropic'])
def trainer(request):
    instance = request.param.__new__(request.param)
    instance.global_steps = 7
    instance.tokenizer = SimpleNamespace(
        eos_token_id=2,
        pad_token_id=0,
        decode=lambda ids, **kwargs: f'text-{int(ids[0])}',
        batch_decode=lambda rows, **kwargs: [f'text-{int(row[0])}' for row in rows],
    )
    return instance


@pytest.fixture
def batch():
    messages = np.empty(3, dtype=object)
    for index in range(3):
        messages[index] = [{'role': 'assistant', 'content': f'action-{index}'}]
    return DataProto.from_dict(
        tensors={
            'input_ids': torch.tensor([[11], [12], [13]]),
            'prompts': torch.tensor([[11], [12], [13]]),
            'responses': torch.tensor([[21], [22], [23]]),
            'token_level_scores': torch.tensor([[.25], [.75], [.25]]),
        },
        non_tensors={
            'episode_ids': np.array([10, 11, 10]),
            'messages_list': messages,
            'data_source': np.array(['CoordSokoban'] * 3),
        },
        meta_info={'metrics': {'CoordSokoban/success': .5}},
    )


@pytest.mark.parametrize('context_mode', ['full', 'single_turn', 'limited_multi_turn'])
@pytest.mark.parametrize('phi_format', [False, True], ids=['legacy', 'phi'])
def test_validation_exports_aligned_rows_without_reference_answers(trainer, batch, context_mode, tmp_path, phi_format):
    if phi_format:
        from ragen.llm_agent.phi import PHI_SPECIAL_TOKENS
        trainer.tokenizer.get_added_vocab = lambda: dict(zip(PHI_SPECIAL_TOKENS, range(5)))
    trainer.config = OmegaConf.create({
        'trainer': {
            'validation_steps': 2,
            'validation_data_dir': str(tmp_path),
            'generations_to_log_to_wandb': {'val': 0},
        },
        'agent_proxy': {'context_window_mode': context_mode},
        'actor_rollout_ref': {'rollout': {'val_kwargs': {'do_sample': False}}},
    })
    trainer.agent_proxy = SimpleNamespace(rollout=lambda data, val: batch)
    trainer.val_reward_fn = lambda data, return_dict: {'reward_tensor': data.batch['token_level_scores']}

    metrics = trainer._validate()

    rows = [json.loads(line) for line in (tmp_path / '7.jsonl').read_text().splitlines()]
    if context_mode == 'full':
        expected_outputs = ([f'assistant\naction-{i}' for i in range(3)] if phi_format
                            else ['text-21', 'text-22', 'text-23'])
        expected_scores = [.25, .75, .25]
    else:
        expected_outputs = [
            '[assistant]\naction-0\n\n[assistant]\naction-2',
            '[assistant]\naction-1',
        ]
        expected_scores = [.25, .75]
    assert [row['output'] for row in rows] == expected_outputs * 2
    assert [row['score'] for row in rows] == expected_scores * 2
    assert [row['reward'] for row in rows] == expected_scores * 2
    assert all(row['input'] == '' and row['gts'] is None and row['step'] == 7 for row in rows)
    assert metrics['val-env/CoordSokoban/success'] == .5
    assert any(key.startswith('val-core/CoordSokoban/reward/mean@') for key in metrics)


@pytest.mark.parametrize('has_ground_truth', [False, True])
def test_rollout_export_uses_verl_ground_truth_contract(trainer, batch, has_ground_truth, tmp_path):
    expected_gts = [None] * 3
    if has_ground_truth:
        expected_gts = ['answer-0', 'answer-1', 'answer-2']
        batch.non_tensor_batch['reward_model'] = np.array(
            [{'ground_truth': answer} for answer in expected_gts], dtype=object)
    timing = {}
    trainer._log_rollout_data(batch, {'success': [False, True, False]}, timing, str(tmp_path))

    rows = [json.loads(line) for line in (tmp_path / '7.jsonl').read_text().splitlines()]
    assert [row['input'] for row in rows] == ['text-11', 'text-12', 'text-13']
    assert [row['output'] for row in rows] == ['text-21', 'text-22', 'text-23']
    assert [row['gts'] for row in rows] == expected_gts
    assert [row['score'] for row in rows] == [.25, .75, .25]
    assert [row['success'] for row in rows] == [False, True, False]
    assert all(row['step'] == 7 for row in rows)
    assert 'dump_rollout_generations' in timing
