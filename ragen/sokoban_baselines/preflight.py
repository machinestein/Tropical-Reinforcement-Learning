"""Compose/validate a new run before creating its output directory or starting Ray."""

import argparse
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .config import validate_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-name', required=True)
    parser.add_argument('--output')
    parser.add_argument('--require-cuda', action='store_true')
    args, overrides = parser.parse_known_args()
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root / 'config'), version_base=None):
        cfg = compose(config_name=args.config_name, overrides=overrides)
        OmegaConf.resolve(cfg)
    validate_config(cfg)
    cfg.data.train_batch_size = cfg.es_manager.train.env_groups * cfg.es_manager.train.group_size
    if args.require_cuda:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != cfg.trainer.n_gpus_per_node:
            raise RuntimeError('All selected CUDA GPUs must be available')
        from .worker import SokobanBaselineWorker  # noqa: F401
        from .actor import DataParallelSokobanBaselineActor  # noqa: F401
    if args.output:
        OmegaConf.save(cfg, args.output)
    print(f'{cfg.trainer.method} configuration preflight passed.')


if __name__ == '__main__':
    main()
