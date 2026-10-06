"""Gemma runtime compatibility and explicit task-specific configuration overrides."""

import os

from omegaconf import OmegaConf, open_dict


def configure_gemma(config):
    with open_dict(config):
        for path in ("actor_rollout_ref.model", "critic.model"):
            model = OmegaConf.select(config, path)
            model.override_config.attn_implementation = "sdpa"
            model.use_remove_padding = False
        config.actor_rollout_ref.model.use_fused_kernels = False
        config.actor_rollout_ref.actor.entropy_from_logits_with_chunking = True
        tags = OmegaConf.select(config, "es_manager.train.env_configs.tags", default=[])
        if any(
            OmegaConf.select(config, f"custom_envs.{tag}.env_type") in {"webshop", "webshop_tropic"}
            for tag in tags
        ):
            config.actor_rollout_ref.actor.entropy_checkpointing = True
            context_limit = os.environ.get("GEMMA_WEBSHOP_MAX_MODEL_LEN")
            if context_limit is not None:
                context_limit = context_limit.strip()
                reserve = int(config.actor_rollout_ref.rollout.response_length)
                if not context_limit.isascii() or not context_limit.isdecimal() or int(context_limit) <= reserve:
                    raise ValueError(
                        f"GEMMA_WEBSHOP_MAX_MODEL_LEN must be an integer greater than response_length ({reserve})"
                    )
                config.actor_rollout_ref.rollout.max_model_len = int(context_limit)
                config.actor_rollout_ref.rollout.max_num_batched_tokens = int(context_limit)
            offload = os.environ.get("GEMMA_WEBSHOP_CPU_OFFLOAD", "false").lower()
            if offload not in {"true", "false", "1", "0"}:
                raise ValueError("GEMMA_WEBSHOP_CPU_OFFLOAD must be true/false or 1/0")
            if offload in {"true", "1"}:
                config.critic.model.fsdp_config.param_offload = True
                config.critic.model.fsdp_config.optimizer_offload = True
            print(
                "Gemma WebShop memory: "
                f"actor_microbatch={config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu}, "
                f"max_model_len={config.actor_rollout_ref.rollout.max_model_len}, "
                f"max_num_batched_tokens={config.actor_rollout_ref.rollout.max_num_batched_tokens}, "
                f"response_length={config.actor_rollout_ref.rollout.response_length}, "
                "entropy_checkpointing=True, "
                f"critic_param_offload={config.critic.model.fsdp_config.param_offload}, "
                f"critic_optimizer_offload={config.critic.model.fsdp_config.optimizer_offload}"
            )
        config.actor_rollout_ref.rollout.layered_summon = True
        config.actor_rollout_ref.actor.ulysses_sequence_parallel_size = 1
        config.actor_rollout_ref.ref.ulysses_sequence_parallel_size = 1
        config.critic.ulysses_sequence_parallel_size = 1
