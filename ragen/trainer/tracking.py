"""RAGEN metric logging with numeric JSON output and native W&B tables."""

import numpy as np
import torch

from verl.utils.tracking import Tracking as VerlTracking


def _metric_value(value, table_type=()):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.ndarray, np.generic)):
        return _metric_value(value.tolist(), table_type)
    if isinstance(value, table_type):
        return _metric_value({
            "_type": "table",
            "columns": value.columns,
            "data": value.data,
        }, table_type)
    if isinstance(value, dict):
        return {key: _metric_value(item, table_type) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_metric_value(item, table_type) for item in value]
    return value


class Tracking(VerlTracking):
    def log(self, data, step, backend=None):
        data = _metric_value(data)
        step = _metric_value(step)
        for name, logger in self.logger.items():
            if backend is not None and name not in backend:
                continue
            backend_data = data
            if name == "file" and "wandb" in self.logger:
                from wandb import Table

                backend_data = _metric_value(data, table_type=Table)
            logger.log(data=backend_data, step=step)
            if name == "file":
                logger.fp.flush()

    def finish(self):
        """Close every backend once, while the process is still healthy.

        The parent destructor closes whatever remains in ``self.logger``; emptying it here
        prevents a second ``wandb.finish`` at interpreter exit, which raced W&B's own
        atexit teardown and could drop the final logged step.
        """
        loggers, self.logger = self.logger, {}
        for name, logger in loggers.items():
            close = getattr(logger, "finish", None)
            if close is None:
                continue
            if name in ("wandb", "vemlp_wandb"):
                close(exit_code=0)
            else:
                close()
