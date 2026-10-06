"""Runtime selection must preserve the existing model and experiment settings."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from train import add_dependency_and_validate_config


def configured(model, validate=True, task="_2_sokoban"):
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "config"),
        version_base=None,
    ):
        config = compose(config_name=task, overrides=[f"model_path={model}"])
    if validate:
        add_dependency_and_validate_config(config)
    return config


@pytest.mark.parametrize(
    "model", ["Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-4-mini-instruct"]
)
def test_existing_models_keep_attention_and_padding(model):
    config = configured(model)
    before = configured(model, validate=False)
    assert (
        config.actor_rollout_ref.model.override_config.get(
            "attn_implementation", "flash_attention_2"
        )
        == "flash_attention_2"
    )
    assert config.actor_rollout_ref.model == before.actor_rollout_ref.model
    assert config.critic.model == before.critic.model
    assert config.actor_rollout_ref.rollout == before.actor_rollout_ref.rollout


@pytest.mark.parametrize("variant", ["E2B", "E4B"])
def test_gemma_selects_native_attention_without_changing_experiment(variant):
    original = configured("Qwen/Qwen2.5-3B-Instruct")
    gemma = configured(f"google/gemma-4-{variant}-it")
    assert gemma.actor_rollout_ref.model.override_config.attn_implementation == "sdpa"
    assert gemma.critic.model.override_config.attn_implementation == "sdpa"
    assert not gemma.actor_rollout_ref.model.use_remove_padding
    assert not gemma.critic.model.use_remove_padding
    assert gemma.actor_rollout_ref.actor.entropy_from_logits_with_chunking
    assert gemma.actor_rollout_ref.rollout.layered_summon
    for path in (
        "algorithm",
        "es_manager",
        "agent_proxy",
        "seed",
        "trainer.total_training_steps",
        "trainer.test_freq",
        "trainer.save_freq",
        "actor_rollout_ref.rollout.temperature",
        "actor_rollout_ref.actor.entropy_coeff",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
    ):
        assert OmegaConf.select(gemma, path) == OmegaConf.select(original, path), path


@pytest.mark.parametrize("variant", ["E2B", "E4B"])
def test_gemma_critic_survives_trainer_dataclass_conversion(variant):
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config import FSDPCriticConfig, FSDPCriticModelCfg

    config = configured(f"google/gemma-4-{variant}-it")
    critic = omega_conf_to_dataclass(config.critic)

    assert isinstance(critic, FSDPCriticConfig)
    assert isinstance(critic.model, FSDPCriticModelCfg)
    assert critic.model.override_config["attn_implementation"] == "sdpa"
    assert not critic.model.use_remove_padding
    assert critic.ppo_mini_batch_size == 32
    assert critic.ppo_micro_batch_size_per_gpu == 1


@pytest.mark.parametrize("task", ["_2_sokoban", "_6_webshop", "_6_webshop_tropic"])
@pytest.mark.parametrize("model", ["Qwen/Qwen2.5-3B-Instruct", "microsoft/Phi-4-mini-instruct", "google/gemma-4-E4B-it"])
def test_entropy_recomputation_is_automatic_only_for_gemma_webshop(task, model):
    before = configured(model, validate=False, task=task)
    after = configured(model, task=task)
    actor = after.actor_rollout_ref.actor
    enabled = model.startswith("google/gemma-") and task.startswith("_6_webshop")
    assert bool(actor.entropy_checkpointing) == enabled
    for path in (
        "algorithm", "es_manager", "agent_proxy", "seed",
        "actor_rollout_ref.actor.entropy_coeff",
        "actor_rollout_ref.actor.ppo_mini_batch_size",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",
        "actor_rollout_ref.rollout.max_model_len",
        "actor_rollout_ref.rollout.response_length",
        "trainer.total_training_steps",
    ):
        assert OmegaConf.select(before, path) == OmegaConf.select(after, path), path
    if not model.startswith("google/gemma-"):
        assert before.actor_rollout_ref.actor == after.actor_rollout_ref.actor


@pytest.mark.parametrize("enabled", ["true", "1"])
@pytest.mark.parametrize("task", ["_6_webshop", "_6_webshop_tropic"])
def test_webshop_cpu_offload_preset_preserves_experiment(monkeypatch, enabled, task):
    from verl.utils.config import omega_conf_to_dataclass

    monkeypatch.delenv("GEMMA_WEBSHOP_CPU_OFFLOAD", raising=False)
    original = configured("google/gemma-4-E4B-it", task=task)
    monkeypatch.setenv("GEMMA_WEBSHOP_CPU_OFFLOAD", enabled)
    actual = configured("google/gemma-4-E4B-it", task=task)
    expected = OmegaConf.create(OmegaConf.to_container(original, resolve=False))
    expected.critic.model.fsdp_config.param_offload = True
    expected.critic.model.fsdp_config.optimizer_offload = True
    assert actual == expected
    critic = omega_conf_to_dataclass(actual.critic)
    assert critic.model.fsdp_config.param_offload
    assert critic.model.fsdp_config.optimizer_offload
    assert actual.actor_rollout_ref.actor.fsdp_config == original.actor_rollout_ref.actor.fsdp_config


@pytest.mark.parametrize("model, task", [
    ("Qwen/Qwen2.5-3B-Instruct", "_6_webshop"),
    ("microsoft/Phi-4-mini-instruct", "_6_webshop"),
    ("google/gemma-4-E4B-it", "_2_sokoban"),
    ("google/gemma-4-E4B-it", "_4_countdown"),
])
def test_webshop_cpu_offload_preset_is_isolated(monkeypatch, model, task):
    monkeypatch.delenv("GEMMA_WEBSHOP_CPU_OFFLOAD", raising=False)
    original = configured(model, task=task)
    monkeypatch.setenv("GEMMA_WEBSHOP_CPU_OFFLOAD", "true")
    assert configured(model, task=task) == original


@pytest.mark.parametrize("disabled", ["false", "0"])
def test_webshop_cpu_offload_preset_can_be_disabled(monkeypatch, disabled):
    monkeypatch.setenv("GEMMA_WEBSHOP_CPU_OFFLOAD", disabled)
    config = configured("google/gemma-4-E4B-it", task="_6_webshop")
    assert not config.critic.model.fsdp_config.param_offload
    assert not config.critic.model.fsdp_config.optimizer_offload
    assert not config.actor_rollout_ref.ref.fsdp_config.param_offload


def test_webshop_cpu_offload_preset_rejects_invalid_value(monkeypatch):
    monkeypatch.setenv("GEMMA_WEBSHOP_CPU_OFFLOAD", "yesplease")
    with pytest.raises(ValueError, match="GEMMA_WEBSHOP_CPU_OFFLOAD"):
        configured("google/gemma-4-E4B-it", task="_6_webshop")


@pytest.mark.parametrize("task", ["_6_webshop", "_6_webshop_tropic"])
@pytest.mark.parametrize("limit", [8192, 4096])
def test_webshop_context_limit_changes_only_explicit_token_budgets(monkeypatch, task, limit):
    monkeypatch.delenv("GEMMA_WEBSHOP_MAX_MODEL_LEN", raising=False)
    monkeypatch.setenv("GEMMA_WEBSHOP_CPU_OFFLOAD", "false")
    original = configured("google/gemma-4-E4B-it", task=task)
    assert original.actor_rollout_ref.rollout.max_model_len == 15000
    monkeypatch.setenv("GEMMA_WEBSHOP_MAX_MODEL_LEN", str(limit))
    actual = configured("google/gemma-4-E4B-it", task=task)
    expected = OmegaConf.create(OmegaConf.to_container(original, resolve=False))
    expected.actor_rollout_ref.rollout.max_model_len = limit
    expected.actor_rollout_ref.rollout.max_num_batched_tokens = limit
    assert actual == expected
    assert not actual.critic.model.fsdp_config.param_offload
    assert not actual.critic.model.fsdp_config.optimizer_offload
    assert actual.actor_rollout_ref.rollout.response_length == 400
    assert actual.agent_proxy.max_turn == 9


@pytest.mark.parametrize("model, task", [
    ("Qwen/Qwen2.5-3B-Instruct", "_6_webshop"),
    ("microsoft/Phi-4-mini-instruct", "_6_webshop"),
    ("google/gemma-4-E4B-it", "_2_sokoban"),
    ("google/gemma-4-E4B-it", "_4_countdown"),
])
def test_webshop_context_limit_is_isolated(monkeypatch, model, task):
    monkeypatch.delenv("GEMMA_WEBSHOP_MAX_MODEL_LEN", raising=False)
    original = configured(model, task=task)
    monkeypatch.setenv("GEMMA_WEBSHOP_MAX_MODEL_LEN", "8192")
    assert configured(model, task=task) == original


@pytest.mark.parametrize("limit", ["", "0", "-1", "400", "8192.5", "invalid"])
def test_webshop_context_limit_reserves_response_space(monkeypatch, limit):
    monkeypatch.setenv("GEMMA_WEBSHOP_MAX_MODEL_LEN", limit)
    with pytest.raises(ValueError, match="GEMMA_WEBSHOP_MAX_MODEL_LEN"):
        configured("google/gemma-4-E4B-it", task="_6_webshop")
