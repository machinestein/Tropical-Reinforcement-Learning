import json
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import wandb

from verl import DataProto
from ragen.trainer.rollout_filter import RewardRolloutFilter, RolloutFilterConfig
from ragen.trainer.tracking import Tracking


@pytest.fixture
def logger(tmp_path, monkeypatch):
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", str(tmp_path / "metrics.jsonl"))
    monkeypatch.setattr(wandb, "init", Mock())
    monkeypatch.setattr(wandb, "log", Mock())
    monkeypatch.setattr(wandb, "finish", Mock())
    instance = Tracking("test", "training", default_backend=["console", "file", "wandb"])
    yield instance
    instance.__del__()
    instance.logger.clear()


def test_logs_tensor_and_numpy_metrics_as_numbers(logger, tmp_path, capsys):
    loss = torch.tensor(1.25, requires_grad=True)
    metrics = {
        "actor/loss": loss,
        "rollout/filter_kept_count": torch.tensor(8),
        "actor/grad_norm": np.float32(2.5),
        "train/batch_size": np.int64(1095),
        "train/empty": np.bool_(False),
        "nested": {"values": (torch.tensor([1., 2.]), np.array([[3, 4]]))},
        "note": None,
    }

    logger.log(metrics, step=np.int64(1))

    record = json.loads((tmp_path / "metrics.jsonl").read_text())
    assert record == {"step": 1, "data": {
        "actor/loss": 1.25,
        "rollout/filter_kept_count": 8,
        "actor/grad_norm": 2.5,
        "train/batch_size": 1095,
        "train/empty": False,
        "nested": {"values": [[1., 2.], [[3, 4]]]},
        "note": None,
    }}
    assert wandb.log.call_args.kwargs == record
    assert "rollout/filter_kept_count:8" in capsys.readouterr().out
    assert metrics["actor/loss"] is loss and loss.requires_grad
    assert isinstance(metrics["nested"]["values"], tuple)


@pytest.mark.parametrize("value", [1.0, 0.9], ids=["ppo_grpo", "snr"])
def test_real_rollout_filter_metrics_and_tables_are_serializable(logger, tmp_path, value):
    batch = DataProto.from_dict(tensors={
        "original_rm_scores": torch.tensor([[-1.], [1.], [0.], [6.]]),
        "loss_mask": torch.ones(4, 1),
    })
    rollout_filter = RewardRolloutFilter(RolloutFilterConfig(
        value=value, filter_type="largest", num_groups=2, group_size=2,
    ))
    _, metrics = rollout_filter.filter(batch)
    reward_matrix = metrics.pop("rollout/_reward_matrix")
    group_std = metrics.pop("rollout/_group_reward_std")
    group_ids = metrics.pop("rollout/_group_ids")
    metrics.pop("rollout/_selected_group_ids")
    reward_table = wandb.Table(columns=["group_0", "group_1"], data=reward_matrix.T.tolist())
    group_table = wandb.Table(columns=["group", "std"], data=[
        [int(group_id), float(std)] for group_id, std in zip(group_ids, group_std)
    ])
    metrics["rollout/reward_table"] = reward_table
    metrics["rollout/group_rv_table"] = group_table
    metrics["rollout/ref_log_prob"] = torch.tensor(-.5)

    logger.log(metrics, step=1)

    record = json.loads((tmp_path / "metrics.jsonl").read_text())
    for key, metric in metrics.items():
        if isinstance(metric, torch.Tensor):
            assert record["data"][key] == pytest.approx(metric.item())
    assert record["data"]["rollout/reward_table"] == {
        "_type": "table", "columns": ["group_0", "group_1"], "data": [[-1., 0.], [1., 6.]],
    }
    assert record["data"]["rollout/group_rv_table"]["data"] == group_table.data
    assert wandb.log.call_args.kwargs["data"]["rollout/reward_table"] is reward_table
    assert wandb.log.call_args.kwargs["data"]["rollout/group_rv_table"] is group_table


@pytest.mark.parametrize("backend", [None, ["file"], ["wandb"], ["console"]])
def test_backend_selection_and_multiple_steps(logger, tmp_path, backend):
    for step in (1, 2):
        logger.log({"actor/loss": torch.tensor(.5)}, step=step, backend=backend)

    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [record["step"] for record in records] == ([1, 2] if backend is None or "file" in backend else [])
    assert wandb.log.call_count == (2 if backend is None or "wandb" in backend else 0)


def test_file_only_logging_does_not_initialize_wandb(tmp_path, monkeypatch):
    monkeypatch.setenv("VERL_FILE_LOGGER_PATH", str(tmp_path / "metrics.jsonl"))
    init = Mock(side_effect=AssertionError("W&B must not initialize for file-only logging"))
    monkeypatch.setattr(wandb, "init", init)
    logger = Tracking("test", "local", default_backend="file")
    try:
        logger.log({"actor/loss": torch.tensor(.5)}, step=1)
        assert json.loads((tmp_path / "metrics.jsonl").read_text())["data"]["actor/loss"] == .5
        init.assert_not_called()
    finally:
        logger.__del__()
        logger.logger.clear()


def test_unsupported_values_are_not_silently_stringified(logger):
    with pytest.raises(TypeError, match="not JSON serializable"):
        logger.log({"bad_metric": object()}, step=1, backend=["file"])


def test_finish_closes_wandb_and_file_once_and_disarms_destructor(logger, tmp_path):
    logger.log({"actor/loss": 1.0}, step=1)
    logger.finish()
    wandb.finish.assert_called_once_with(exit_code=0)
    assert logger.logger == {}
    assert json.loads((tmp_path / "metrics.jsonl").read_text())["step"] == 1
    logger.__del__()  
    wandb.finish.assert_called_once()
