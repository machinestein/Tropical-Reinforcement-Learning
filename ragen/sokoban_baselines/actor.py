"""One full-batch update; MaxRL or GRPO plus thought-only preference gradients."""

from contextlib import contextmanager

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from ragen.workers.actor.dp_actor import DataParallelPPOActor
from .core import policy_loss, surgical_loss


class ShardEMA:
    """CPU copies of local FSDP1 shards, not an additional full GPU model.

    Invalidate inference's cached unsharded weights before/after a swap. FSDP1
    retains the root's full parameter after a no-grad forward; copying the local
    shard alone would silently evaluate the wrong reference policy.
    Supported with FULL_SHARD/HYBRID_SHARD, use_orig_params=False and no LoRA.
    """
    def __init__(self, module):
        self.values = {name: p.detach().float().cpu().clone() for name, p in module.named_parameters()}

    @staticmethod
    def flush(module):
        if isinstance(module, FSDP):
            from torch.distributed.fsdp._runtime_utils import _reshard
            for wrapper in FSDP.fsdp_modules(module):
                handle = wrapper._handle
                if handle is not None and handle.uses_sharded_strategy:
                    _reshard(wrapper, handle, True)

    @torch.no_grad()
    def update(self, module, alpha):
        for name, p in module.named_parameters():
            self.values[name].mul_(alpha).add_(p.detach().float().cpu(), alpha=1 - alpha)

    @contextmanager
    def applied(self, module):
        self.flush(module)
        current = {name: p.detach().cpu().clone() for name, p in module.named_parameters()}
        try:
            with torch.no_grad():
                for name, p in module.named_parameters():
                    p.copy_(self.values[name].to(device=p.device, dtype=p.dtype))
            yield
        finally:
            self.flush(module)
            with torch.no_grad():
                for name, p in module.named_parameters():
                    p.copy_(current[name].to(p.device))


class DataParallelSokobanBaselineActor(DataParallelPPOActor):
    def __init__(self, config, actor_module, actor_optimizer):
        super().__init__(config, actor_module, actor_optimizer)
        for module in actor_module.modules():
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                module.p = 0.0
            if hasattr(module, 'attention_dropout') and isinstance(module.attention_dropout, (int, float)):
                module.attention_dropout = 0.0

    def update_policy(self, data):
        method = data.meta_info['baseline_method']
        if method not in ('maxrl', 'tstar') or self.config.ppo_epochs != 1:
            raise ValueError("Sokoban baselines require exactly one update per generated batch")
        self.actor_module.train()  
        device = next(self.actor_module.parameters()).device
        batch = data.batch
        n_rl = data.meta_info['baseline_rl_rows_per_rank']
        n_pairs = data.meta_info['baseline_pairs_per_rank']
        world = data.meta_info['baseline_dp_size']
        if len(batch) != n_rl + 2 * n_pairs:
            raise ValueError("Data parallel dispatch split a preference pair")
        tstar = method == 'tstar'
        if tstar and not hasattr(self, 'ema'):
            self.ema = ShardEMA(self.actor_module)
        metrics = {'actor/policy_loss': 0.0, 'actor/surgical_loss': 0.0,
                   'actor/optimizer_steps': 0, 'actor/grad_norm': 0.0}
        if data.meta_info.get('baseline_skip_update', False):
            if tstar:
                self.ema.update(self.actor_module, data.meta_info['tstar_ema_alpha'])
            return metrics

        def score(micro):
            _, log_probs = self._forward_micro_batch(micro, temperature=1.0, calculate_entropy=False)
            return log_probs

        def summed(log_probs, micro):
            return log_probs.masked_fill(~micro['response_mask'].bool(), 0.0).sum(-1)

        def backward(loss):
            finite = torch.isfinite(loss.detach()).to(torch.int32)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(finite, op=torch.distributed.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError("Non-finite Sokoban baseline loss")
            (world * loss).backward()  

        refs = []
        if n_pairs:
            self.actor_module.eval()
            with self.ema.applied(self.actor_module), torch.no_grad():
                for i in range(n_rl, len(batch)):
                    micro = batch[i:i + 1].to(device)
                    refs.append(summed(score(micro), micro).detach())
            self.actor_module.train()
        self.actor_optimizer.zero_grad()
        for micro in batch[:n_rl].split(self.config.ppo_micro_batch_size_per_gpu):
            micro = micro.to(device)
            log_probs = score(micro)
            loss = policy_loss(log_probs, micro['response_mask'], micro['baseline_advantages'],
                               micro['baseline_weights'], micro['old_log_probs'] if tstar else None,
                               clip=self.config.clip_ratio_low)
            backward(loss)
            metrics['actor/policy_loss'] += world * float(loss.detach())

        for pair in range(n_pairs):
            idx = n_rl + 2 * pair
            selected = batch[idx:idx + 2].to(device)
            current = summed(selected['old_log_probs'], selected)
            beta, coefficient = data.meta_info['tstar_beta'], data.meta_info['tstar_surgical_weight']
            ref_plus, ref_minus = refs[2 * pair], refs[2 * pair + 1]
            gap = (current[0] - current[1]) - (ref_plus - ref_minus)
            weight = selected['pair_weights'][0] * coefficient
            derivative = (-beta * torch.sigmoid(-beta * gap) * weight).detach()
            value = surgical_loss(current[0], current[1], ref_plus, ref_minus, beta) * weight
            metrics['actor/surgical_loss'] += world * float(value.detach())
            for side, sign in ((0, 1), (1, -1)):
                micro = selected[side:side + 1]
                backward(sign * derivative * summed(score(micro), micro).sum())
        norm = self._optimizer_step()
        if not torch.isfinite(norm):
            raise FloatingPointError("Non-finite Sokoban baseline gradient norm")
        metrics.update({'actor/grad_norm': float(norm.detach()), 'actor/optimizer_steps': 1})
        self.actor_optimizer.zero_grad()
        if tstar:
            self.ema.update(self.actor_module, data.meta_info['tstar_ema_alpha'])
        return metrics
