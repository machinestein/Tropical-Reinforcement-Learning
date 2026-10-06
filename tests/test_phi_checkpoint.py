"""Native Phi checkpoints must save without packaging Transformers as Hub code."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, Phi3Config

from ragen.workers.phi_checkpoint import prepare_native_phi_checkpoint


def phi_model(kind="actor"):
    config = Phi3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
        original_max_position_embeddings=32, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, num_labels=1,
        auto_map={
            "AutoConfig": "configuration_phi3.Phi3Config",
            "AutoModelForCausalLM": "modeling_phi3.Phi3ForCausalLM",
            "AutoTokenizer": "Xenova/gpt-4o",
        },
    )
    factory = AutoModelForCausalLM if kind == "actor" else AutoModelForTokenClassification
    return factory.from_config(config, trust_remote_code=False, attn_implementation="eager")


@pytest.mark.parametrize("kind", ["actor", "critic"])
def test_native_phi_checkpoint_save_and_reload(kind, tmp_path, monkeypatch):
    from verl.utils.checkpoint import checkpoint_manager as base
    from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(base, "get_device_name", lambda: "cpu")
    model = phi_model(kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    next(model.parameters()).sum().backward()
    optimizer.step()
    scheduler.step()
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    before = copy.deepcopy(model.config.to_dict())

    assert prepare_native_phi_checkpoint(model)
    before.pop("auto_map")
    assert model.config.to_dict() == before
    manager = FSDPCheckpointManager(model, optimizer, scheduler)
    manager.save_checkpoint(str(tmp_path), global_step=100)

    saved_config = json.loads((tmp_path / "huggingface/config.json").read_text())
    assert "auto_map" not in saved_config
    assert not (tmp_path / "huggingface/modeling_phi3.py").exists()
    assert (tmp_path / "fsdp_config.json").exists()
    config = AutoConfig.from_pretrained(tmp_path / "huggingface", trust_remote_code=False, local_files_only=True)
    factory = AutoModelForCausalLM if kind == "actor" else AutoModelForTokenClassification
    restored = factory.from_config(config, trust_remote_code=False)
    restored.load_state_dict(torch.load(tmp_path / "model_world_size_1_rank_0.pt", weights_only=False))
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    with torch.no_grad():
        next(model.parameters()).zero_()
    optimizer.state.clear()
    scheduler.step()
    manager.load_checkpoint(str(tmp_path))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    assert optimizer.state
    assert scheduler.last_epoch == 1


@pytest.mark.parametrize("wrapper", ["fsdp1", "fsdp2"])
def test_phi_config_is_prepared_through_fsdp_wrapping(wrapper):
    model = phi_model()
    if wrapper == "fsdp1":
        wrapped = SimpleNamespace(_fsdp_wrapped_module=model)
    else:
        from torch.distributed.fsdp import FSDPModule
        model.__class__ = type(f"FSDP{type(model).__name__}", (FSDPModule, type(model)), {})
        wrapped = model
    assert prepare_native_phi_checkpoint(wrapped)
    assert not hasattr(model.config, "auto_map")
    assert not prepare_native_phi_checkpoint(wrapped)  


def test_qwen_config_and_custom_phi_classes_are_unchanged():
    from transformers import Qwen2Config

    qwen_config = Qwen2Config(auto_map={"AutoModelForCausalLM": "custom.Qwen"})
    before = copy.deepcopy(qwen_config.to_dict())
    assert not prepare_native_phi_checkpoint(SimpleNamespace(config=qwen_config))
    assert qwen_config.to_dict() == before

    model = phi_model()
    before = copy.deepcopy(model.config.to_dict())
    model.__class__ = type("CustomPhi", (type(model),), {"__module__": "transformers_modules.custom"})
    assert not prepare_native_phi_checkpoint(model)
    assert model.config.to_dict() == before


@pytest.mark.parametrize("worker_name", ["ActorRolloutRefWorker", "AsyncActorRolloutRefWorker", "CriticWorker"])
def test_worker_initialization_prepares_phi_checkpoints(worker_name, monkeypatch):
    from ragen.workers import fsdp_workers as workers
    from ragen.workers.actor import dp_actor

    worker_class = getattr(workers, worker_name)
    worker = object.__new__(worker_class)
    model = phi_model("critic" if worker_name == "CriticWorker" else "actor")
    worker._is_actor, worker._is_rollout = True, True
    worker.config = OmegaConf.create({"actor": {}})
    worker.actor_module_fsdp, worker.actor_optimizer = model, None
    monkeypatch.setattr(worker_class.__bases__[0], "init_model",
                        lambda self: setattr(self, "checkpoint_manager", SimpleNamespace(model=model)))
    monkeypatch.setattr(dp_actor, "DataParallelPPOActor", lambda **kwargs: SimpleNamespace(**kwargs))
    worker.init_model()
    assert not hasattr(model.config, "auto_map")
