"""Opt-in worker; existing PPO/TROPIC workers retain their behavior."""

from verl.single_controller.base.decorator import Dispatch, register, make_nd_compute_dataproto_dispatch_fn
from verl.workers.fsdp_workers import ActorRolloutRefWorker as VerlActorRolloutRefWorker
from ragen.workers.fsdp_workers import ActorRolloutRefWorker


class SokobanBaselineWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if self._is_actor:
            from .actor import DataParallelSokobanBaselineActor
            self.actor = DataParallelSokobanBaselineActor(self.config.actor, self.actor_module_fsdp,
                                                        self.actor_optimizer)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name='actor'))
    def update_actor(self, data):
        return VerlActorRolloutRefWorker.update_actor(self, data)
