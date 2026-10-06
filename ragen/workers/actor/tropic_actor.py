"""Full-batch verified-path updates on the existing FSDP actor."""

import torch
from ragen.tropic.batch import verified_path_loss
from .dp_actor import DataParallelPPOActor


class DataParallelTropicActor(DataParallelPPOActor):
    def update_policy(self, data, skip_optimizer_step=False):
        if self.config.use_kl_loss or self.config.entropy_coeff:
            raise ValueError("The initial TROPIC update implements unregularized verified-path MLE")
        self.actor_module.eval()  
        device = next(self.actor_module.parameters()).device
        keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'response_mask', 'tropic_weights']
        batch = data.select(batch_keys=keys).batch
        dp_size = int(data.meta_info['tropic_dp_size'])
        metrics = {'actor/verified_nll': [], 'actor/grad_norm': []}
        for _ in range(self.config.ppo_epochs):
            self.actor_optimizer.zero_grad()
            total_loss = 0.0
            for micro in batch.split(self.config.ppo_micro_batch_size_per_gpu):
                micro = micro.to(device)
                _, log_probs = self._forward_micro_batch(micro, temperature=1.0, calculate_entropy=False)
                loss = dp_size * verified_path_loss(log_probs, micro['response_mask'], micro['tropic_weights'])
                finite = torch.isfinite(loss.detach()).to(torch.int32)
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError("Non-finite verified-path loss")
                loss.backward()
                total_loss += float(loss.detach())
            norm = self._optimizer_step(skip_step=skip_optimizer_step)
            if not torch.isfinite(norm):
                raise FloatingPointError("Non-finite verified-path gradient norm")
            metrics['actor/verified_nll'].append(total_loss)
            metrics['actor/grad_norm'].append(float(norm.detach()))
        self.actor_optimizer.zero_grad()
        return metrics
