"""Separate training loop; ordinary RAGEN validation always samples fresh roots."""

import json
from pathlib import Path
import time

import numpy as np
from omegaconf import OmegaConf
from verl.utils.metric import reduce_metrics

from ragen.trainer.agent_trainer import RayAgentTrainer
from ragen.trainer.tracking import Tracking
from .batch import training_batch
from .collector import BaselineCollector


class RaySokobanBaselineTrainer(RayAgentTrainer):
    def init_agent_proxy(self):
        super().init_agent_proxy()
        self.collector = BaselineCollector(self.config, self.agent_proxy, self.actor_rollout_wg, self.tokenizer)

    def fit(self):
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))
        self.global_steps = 0
        output_dir = Path(self.config.trainer.default_local_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        method = self.config.trainer.method
        started = time.perf_counter()
        try:
            if self.config.trainer.val_before_train:
                logger.log(self._validate(), step=0)
            if self.config.trainer.val_only:
                return
            for step in range(1, self.total_training_steps + 1):
                self.global_steps = step
                begin = time.perf_counter()
                rows, advantages, pairs, metrics = self.collector.collect()
                metrics['timing_s/rollout'] = time.perf_counter() - begin
                batch = training_batch(rows, advantages, pairs, self.tokenizer.pad_token_id,
                                       self.actor_rollout_wg.world_size,
                                       self.config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu, method)
                if method == 'tstar':
                    batch.meta_info.update(tstar_beta=self.config.tstar.beta,
                                           tstar_surgical_weight=self.config.tstar.surgical_weight,
                                           tstar_ema_alpha=self.config.tstar.ema_alpha)
                    batch = batch.union(self.actor_rollout_wg.compute_log_prob(batch))
                batch.meta_info['baseline_skip_update'] = not np.any(advantages) and not pairs
                update_start = time.perf_counter()
                output = self.actor_rollout_wg.update_actor(batch)
                metrics.update(reduce_metrics(output.meta_info['metrics']))
                metrics['timing_s/update_actor'] = time.perf_counter() - update_start
                with (output_dir / f'{method}_events.jsonl').open('a') as stream:
                    for event in self.collector.events:
                        stream.write(json.dumps({'step': step, **event}) + '\n')
                freq = self.config.trainer.test_freq
                if freq > 0 and (step % freq == 0 or step == self.total_training_steps):
                    metrics.update(self._validate())
                freq = self.config.trainer.save_freq
                if freq > 0 and (step % freq == 0 or step == self.total_training_steps):
                    self._save_checkpoint()
                metrics['timing_s/step'] = time.perf_counter() - begin
                metrics['timing_s/elapsed'] = time.perf_counter() - started
                logger.log(metrics, step=step)
        finally:
            self.agent_proxy.train_es_manager.close()
            self.agent_proxy.val_es_manager.close()
            if hasattr(logger, 'finish'):
                logger.finish()
