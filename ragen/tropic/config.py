"""Fail early on unsupported TROPIC collection and update configurations."""


def validate_tropic_config(config):
    agent = config.agent_proxy
    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    options = config.tropic
    if not isinstance(options.get('successful_fragments_only', False), bool):
        raise ValueError("successful_fragments_only must be a boolean")
    if agent.context_window_mode != 'single_turn' or agent.max_context_window != 1:
        raise ValueError("TROPIC requires single_turn context with max_context_window=1")
    if agent.max_actions_per_turn != 1 or not agent.reject_extra_actions or not agent.terminate_on_invalid_action:
        raise ValueError("TROPIC requires one action per turn and termination on malformed actions")
    if config.critic.enable or actor.use_ref or actor.use_kl_loss or actor.entropy_coeff:
        raise ValueError("TROPIC uses no critic, reference policy, KL loss or entropy bonus")
    if config.algorithm.use_kl_in_reward or config.algorithm.bi_level_gae:
        raise ValueError("TROPIC does not use advantage estimation or reward KL")
    if actor.ulysses_sequence_parallel_size != 1 or actor.strategy != 'fsdp':
        raise ValueError("The first TROPIC implementation requires FSDP and sequence parallel size 1")
    if rollout.temperature != 1 or rollout.top_p != 1 or rollout.top_k not in (-1, 0):
        raise ValueError("TROPIC accessibility scoring requires temperature=1, top_p=1 and no top_k truncation")
    if config.trainer.gradient_analysis_mode or config.trainer.gradient_analysis_only:
        raise ValueError("PPO gradient decomposition is not a TROPIC diagnostic")
    if config.trainer.default_hdfs_dir is not None:
        raise ValueError("TROPIC checkpoints currently support local storage only")
    if config.collapse_detection.compute_freq < 1:
        raise ValueError("collapse_detection.compute_freq must be positive")
    if rollout.rollout_filter_value != 1.0:
        raise ValueError("TROPIC does not apply PPO rollout filtering")
    train = config.es_manager.train
    tags = list(train.env_configs.tags)
    supported = ('sokoban', 'lean_tropic', 'webshop_tropic', 'sudoku_tropic', 'countdown_step', 'frozen_lake_tropic')
    if len(tags) != 1 or config.custom_envs[tags[0]].env_type not in supported:
        raise ValueError("TROPIC supports one Sokoban, LeanTropic, WebShopTropic, SudokuTropic, CountdownStep or FrozenLakeTropic environment tag")
    task = config.custom_envs[tags[0]]
    if options.get('successful_fragments_only', False) and task.env_type not in ('sokoban', 'countdown_step'):
        raise ValueError("The successful-fragments-only ablation currently supports Sokoban and CountdownStep")
    if task.env_type in ('sokoban', 'sudoku_tropic', 'countdown_step', 'frozen_lake_tropic') and task.env_config.get('render_mode', 'text') != 'text':
        raise ValueError("TROPIC currently supports text observations only")
    if task.env_type == 'frozen_lake_tropic' and task.env_config.get('success_rate', 1.0) != 1.0:
        raise ValueError("FrozenLake TROPIC requires deterministic movement (success_rate=1)")
    if agent.max_turn < task.max_actions_per_traj or task.env_config.max_steps < task.max_actions_per_traj:
        raise ValueError("Turn and simulator limits must cover the full action budget")
    if task.env_type == 'webshop_tropic' and task.env_config.get('observation_mode', 'text') != 'text':
        raise ValueError("WebShop TROPIC requires text observations")
    if train.seed_pool_size < train.env_groups or config.seed.train is None:
        raise ValueError("TROPIC needs a fixed recurring training seed pool")
    for name in ('waves_per_iteration', 'max_edges_per_problem', 'max_solutions_per_problem',
                 'top_l', 'basis_size', 'max_candidates_per_refresh', 'rescore_batch_size'):
        if int(options[name]) < 1:
            raise ValueError(f"tropic.{name} must be positive")
    if options.frontier_eta < 0 or actor.ppo_epochs < 1 or actor.ppo_micro_batch_size_per_gpu < 1:
        raise ValueError("Invalid exploration or update settings")
    if len(str(config.system.CUDA_VISIBLE_DEVICES).split(',')) != config.trainer.n_gpus_per_node:
        raise ValueError("CUDA_VISIBLE_DEVICES and n_gpus_per_node disagree")
