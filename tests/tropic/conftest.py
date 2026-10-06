from pathlib import Path
from types import SimpleNamespace
import re

from hydra import compose, initialize_config_dir
import pytest
import torch


@pytest.fixture
def config():
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[2] / 'config'), version_base=None):
        cfg = compose(config_name='_2_sokoban_tropic')
    cfg.es_manager.train.env_groups = 1
    cfg.es_manager.train.group_size = 2
    cfg.es_manager.train.env_configs.n_groups = [1]
    cfg.es_manager.train.seed_pool_size = 2
    cfg.es_manager.val.env_groups = 1
    cfg.es_manager.val.group_size = 1
    cfg.es_manager.val.env_configs.n_groups = [1]
    cfg.custom_envs.CoordSokoban.env_config.dim_x = 5
    cfg.custom_envs.CoordSokoban.env_config.dim_y = 5
    cfg.custom_envs.CoordSokoban.env_config.search_depth = 10
    cfg.custom_envs.CoordSokoban.max_actions_per_traj = 5
    cfg.agent_proxy.max_turn = 5
    cfg.actor_rollout_ref.rollout.max_model_len = 10000
    cfg.tropic.waves_per_iteration = 1
    return cfg


class CharacterTokenizer:
    """Deterministic chat tokenizer for tests that never downloads model weights."""

    name_or_path = 'qwen-test'
    pad_token_id = 0
    eos_token_id = 2
    special = {'<|im_start|>': 1, '<|im_end|>': 2}

    def encode(self, text, **kwargs):
        parts = re.split(r'(<\|im_start\|>|<\|im_end\|>)', text)
        return [token for part in parts for token in
                ([self.special[part]] if part in self.special else [ord(c) + 10 for c in part])]

    def decode(self, ids, skip_special_tokens=False, **kwargs):
        special = {0: '', 1: '<|im_start|>', 2: '<|im_end|>'}
        return ''.join(('' if skip_special_tokens else special[int(i)]) if int(i) in special
                       else chr(int(i) - 10) for i in ids)

    def batch_decode(self, ids, **kwargs):
        return [self.decode(row, **kwargs) for row in ids]

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            return {'input_ids': self.encode(texts)}
        rows = [self.encode(t) for t in texts]
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            ids[i, -len(row):] = torch.tensor(row)
            mask[i, -len(row):] = 1
        return SimpleNamespace(input_ids=ids, attention_mask=mask)

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=False, **kwargs):
        text = ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
        if add_generation_prompt:
            text += '<|im_start|>assistant\n'
        return self.encode(text) if tokenize else text
