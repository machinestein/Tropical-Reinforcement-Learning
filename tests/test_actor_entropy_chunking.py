"""The actor must honour entropy_from_logits_with_chunking on the padded (no-rmpad) path."""

from omegaconf import OmegaConf
import pytest
import torch

from verl.utils import torch_functional as verl_F
from ragen.workers.actor.dp_actor import DataParallelPPOActor


def make_actor(chunked, checkpointed=False):
    actor = DataParallelPPOActor.__new__(DataParallelPPOActor)
    actor.config = OmegaConf.create({"entropy_from_logits_with_chunking": chunked, "use_torch_compile": False,
                                     "use_remove_padding": False, "ulysses_sequence_parallel_size": 1})
    actor.entropy_chunked = chunked
    actor.entropy_checkpointing = checkpointed
    return actor


@pytest.mark.parametrize("chunked, checkpointed", [(False, False), (True, False), (True, True), (False, True)])
def test_padded_entropy_matches_reference(chunked, checkpointed):
    torch.manual_seed(0)
    logits = torch.randn(3, 5, 4099, dtype=torch.float32) * 3  
    entropy = make_actor(chunked, checkpointed)._padded_entropy(logits)
    assert entropy.shape == (3, 5)
    torch.testing.assert_close(entropy, verl_F.entropy_from_logits(logits), rtol=1e-5, atol=1e-5)


def test_flag_selects_chunked_function_for_rmpad_path():
    cfg = OmegaConf.create({"entropy_from_logits_with_chunking": True, "use_torch_compile": True,
                            "use_remove_padding": False, "ulysses_sequence_parallel_size": 1})
    actor = DataParallelPPOActor(cfg, actor_module=torch.nn.Identity())
    assert actor.entropy_chunked and actor.compute_entropy_from_logits is verl_F.entropy_from_logits_with_chunking
    cfg.entropy_from_logits_with_chunking = False
    actor = DataParallelPPOActor(cfg, actor_module=torch.nn.Identity())
    assert not actor.entropy_chunked


def test_checkpointed_function_selected_even_when_compile_enabled():
    from ragen.workers.actor.entropy import entropy_from_logits_recomputed

    cfg = OmegaConf.create({"entropy_from_logits_with_chunking": True, "entropy_checkpointing": True,
                            "use_torch_compile": True, "use_remove_padding": False,
                            "ulysses_sequence_parallel_size": 1})
    actor = DataParallelPPOActor(cfg, actor_module=torch.nn.Identity())
    assert actor.compute_entropy_from_logits is entropy_from_logits_recomputed
