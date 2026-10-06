"""Check batching changes through real worker loops with a tiny CPU model."""

from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from verl import DataProto
from ragen.workers.actor.dp_actor import DataParallelPPOActor
from ragen.workers.actor.tropic_actor import DataParallelTropicActor
from ragen.workers.critic.dp_critic import DataParallelPPOCritic


@pytest.fixture
def batch(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    responses = torch.arange(1, 33).reshape(8, 4)
    lengths = torch.tensor([1, 4, 2, 3, 4, 2, 1, 3])
    mask = (torch.arange(4)[None, :] < lengths[:, None]).double()
    return DataProto.from_dict(tensors={
        "responses": responses,
        "input_ids": torch.cat([torch.zeros(8, 1, dtype=torch.long), responses], dim=1),
        "attention_mask": torch.cat([torch.ones(8, 1), mask], dim=1),
        "position_ids": torch.arange(5).expand(8, -1),
        "response_mask": mask,
        "old_log_probs": torch.full((8, 4), -.3, dtype=torch.float64),
        "advantages": torch.linspace(-2, 3, 32, dtype=torch.float64).reshape(8, 4),
        "values": torch.zeros(8, 4, dtype=torch.float64),
        "returns": torch.linspace(-1, 2, 32, dtype=torch.float64).reshape(8, 4),
        "tropic_weights": torch.tensor([.1, .2, .3, .15, .25, .1, 0, 0], dtype=torch.float64),
    }, meta_info={"temperature": 1.0, "tropic_dp_size": 1})


def run_updates(worker_class, batch, microbatch, loss_mode, entropy_coeff=0.0):
    worker = worker_class.__new__(worker_class)
    worker.config = OmegaConf.create({
        "ppo_mini_batch_size": 4,  
        "ppo_micro_batch_size_per_gpu": microbatch,
        "ppo_epochs": 2,
        "use_dynamic_bsz": False,
        "use_kl_loss": False,
        "entropy_coeff": entropy_coeff,
        "loss_agg_mode": loss_mode,
        "clip_ratio": .2,
        "clip_ratio_low": .2,
        "clip_ratio_high": .28,
        "cliprange_value": .2,
        "grad_clip": 100.,
    })
    model = torch.nn.Linear(1, 1, dtype=torch.float64)
    with torch.no_grad():
        model.weight.fill_(.2)
        model.bias.fill_(-.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    result = SimpleNamespace(gradients=[], forward_calls=0)
    original_step = optimizer.step

    def record_step(*args, **kwargs):
        result.gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]).clone())
        return original_step(*args, **kwargs)

    optimizer.step = record_step

    def forward(micro_batch, **kwargs):
        result.forward_calls += 1
        predictions = model(micro_batch["responses"].double().unsqueeze(-1) / 10).squeeze(-1)
        if worker_class is DataParallelPPOCritic:
            return predictions
        return predictions.square(), predictions

    worker._forward_micro_batch = forward
    if worker_class is DataParallelPPOCritic:
        worker.critic_module, worker.critic_optimizer = model, optimizer
        worker.update_critic(batch)
    else:
        worker.actor_module, worker.actor_optimizer = model, optimizer
        worker.update_policy(batch)
    result.parameters = torch.cat([p.detach().flatten() for p in model.parameters()])
    return result


@pytest.mark.parametrize("microbatch", [2, 4])
@pytest.mark.parametrize(("worker_class", "baseline_mode", "entropy"), [
    (DataParallelPPOActor, "token-mean", 0.),
    (DataParallelPPOActor, "token-mean", .001),
    (DataParallelPPOActor, "seq-mean-token-mean", 0.),
    (DataParallelPPOCritic, "token-mean", 0.),
    (DataParallelTropicActor, "token-mean", 0.),
], ids=["ppo_snr", "original_ppo", "grpo", "critic", "tropic"])
def test_larger_microbatches_preserve_gradients_and_optimizer_steps(
    batch, microbatch, worker_class, baseline_mode, entropy,
):
    baseline = run_updates(worker_class, batch, 1, baseline_mode, entropy)
    larger = run_updates(worker_class, batch, microbatch, "seq-mean-token-mean", entropy)

    assert larger.forward_calls * microbatch == baseline.forward_calls
    assert len(larger.gradients) == len(baseline.gradients)
    for actual, expected in zip(larger.gradients, baseline.gradients):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(larger.parameters, baseline.parameters)


@pytest.mark.parametrize("worker_class", [DataParallelPPOActor, DataParallelPPOCritic])
def test_naive_token_mean_microbatch_change_would_change_response_weighting(batch, worker_class):
    baseline = run_updates(worker_class, batch, 1, "token-mean")
    naive = run_updates(worker_class, batch, 4, "token-mean")
    assert not torch.allclose(naive.gradients[0], baseline.gradients[0])
