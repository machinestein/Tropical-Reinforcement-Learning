"""Validate these opt-in methods without changing legacy config validation."""

import math
from ragen.llm_agent.phi import is_phi_model_path
from ragen.llm_agent.gemma import is_gemma_model_path


def validate_config(config):
    method = config.trainer.method
    actor, rollout = config.actor_rollout_ref.actor, config.actor_rollout_ref.rollout
    if method not in ('maxrl', 'tstar'):
        raise ValueError("Unknown Sokoban baseline")
    if config.critic.enable or actor.use_ref or actor.use_kl_loss or actor.entropy_coeff:
        raise ValueError("These baselines use no critic, frozen reference, KL penalty or entropy bonus")
    if config.algorithm.use_kl_in_reward or config.algorithm.bi_level_gae:
        raise ValueError("These baselines use binary terminal success without reward KL or GAE")
    if actor.strategy != 'fsdp' or actor.ulysses_sequence_parallel_size != 1:
        raise ValueError("Sokoban baselines require FSDP1 and sequence parallel size 1")
    if actor.fsdp_config.get('use_orig_params', False) or config.lora.rank:
        raise ValueError("Sokoban baselines currently require flat FSDP parameters and full-weight training")
    if actor.ppo_epochs != 1 or actor.ppo_micro_batch_size_per_gpu < 1 or actor.use_dynamic_bsz:
        raise ValueError("Use one full-batch optimizer update, positive microbatch size, and fixed microbatches")
    if rollout.temperature != 1 or rollout.top_p != 1 or rollout.top_k not in (-1, 0) or rollout.n != 1:
        raise ValueError("On-policy scoring requires temperature=1, top_p=1, no top_k truncation and n=1")
    if rollout.rollout_filter_value != 1 or not rollout.rollout_filter_include_zero:
        raise ValueError("These baselines do not filter sampled trajectory groups")
    if config.agent_proxy.context_window_mode != 'full' or not config.agent_proxy.enable_think:
        raise ValueError("Use the original full-history Sokoban protocol with reasoning enabled")
    if (not any(name in config.model_path.lower() for name in ('qwen', 'llama-3'))
            and not is_phi_model_path(config.model_path) and not is_gemma_model_path(config.model_path)):
        raise ValueError("Validation response masks support Qwen, Llama-3, Phi-4-mini-instruct and Gemma-4")
    for mode in ('train', 'val'):
        env = config.es_manager[mode]
        tags = list(env.env_configs.tags)
        if len(tags) != 1 or config.custom_envs[tags[0]].env_type != 'sokoban':
            raise ValueError("These launchers support one Sokoban environment tag only")
        if list(env.env_configs.n_groups) != [env.env_groups] or env.env_groups < 1 or env.group_size < 1:
            raise ValueError("Environment groups and tag counts must agree and be positive")
    if config.es_manager.train.group_size < 2:
        raise ValueError("At least two trajectories per training problem are required")
    if config.trainer.resume_mode != 'disable':
        raise ValueError("Baseline training resume is not implemented; use a fresh experiment (checkpoints are evaluable)")
    if config.trainer.default_hdfs_dir is not None:
        raise ValueError("These baselines currently support local checkpoints only")
    if config.trainer.gradient_analysis_mode or config.trainer.gradient_analysis_only:
        raise ValueError("PPO gradient decomposition is not supported by the new update loop")
    devices = str(config.system.CUDA_VISIBLE_DEVICES).split(',')
    if len(devices) != config.trainer.n_gpus_per_node or len(set(devices)) != len(devices):
        raise ValueError("CUDA_VISIBLE_DEVICES and n_gpus_per_node must agree without duplicate GPUs")
    if config.sokoban_baseline.score_batch_size < 1:
        raise ValueError("score_batch_size must be positive")
    if method == 'maxrl':
        if not 0 < config.maxrl.epsilon <= .001 or actor.loss_agg_mode != 'token-mean':
            raise ValueError("MaxRL requires a small positive epsilon and token-mean loss")
    else:
        tstar = config.tstar
        for key in ('kl_threshold', 'gamma', 'divergence_threshold', 'surgical_weight', 'beta', 'ema_alpha'):
            if not math.isfinite(tstar[key]):
                raise ValueError(f"Non-finite tstar.{key}")
        if (tstar.kl_samples < 1 or tstar.kl_threshold <= 0 or not 0 <= tstar.gamma <= 1 or
                tstar.divergence_threshold < 0 or tstar.surgical_weight < 0 or tstar.beta <= 0 or
                not 0 <= tstar.ema_alpha < 1 or tstar.graft_batch_size < 0):
            raise ValueError("Invalid T-STAR hyperparameters")
        if actor.clip_ratio_low != actor.clip_ratio_high or actor.loss_agg_mode != 'seq-mean-token-mean':
            raise ValueError("T-STAR uses symmetric GRPO clipping and trajectory-normalized token loss")
