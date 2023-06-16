import torch

import ai

from aic.data.dataset.worker import DataWorker


def data_iters(cfg, n_val_samples, val_bs, device='cuda', include_ply=False, n_train_workers=6):
    return (
        train_data_iter(cfg, device, include_ply, n_train_workers),
        val_data_iter(cfg, n_val_samples, val_bs, device, include_ply),
    )


def train_data_iter(cfg, device='cuda', include_ply=False, n_workers=6):
    return _data_iter(cfg, cfg.train.bs, device, include_ply, n_workers, train=True)


def val_data_iter(cfg, n_samples, bs, device='cuda', include_ply=False):
    it = iter(_data_iter(cfg, bs, device, include_ply, n_workers=0, train=False))
    n_batches = n_samples // bs
    return [next(it) for _ in range(n_batches)]


def _data_iter(cfg, bs, device, include_ply, n_workers, train):
    return ai.data.DataIterator(
        torch.utils.data.DataLoader(
            DataWorker(
                cfg.data.db,
                cfg.task,
                bs,
                cfg.data.drop_chance_by_elo_bin,
                cfg.data.rand_trim_start,
                cfg.data.rand_trim_end,
                include_ply,
                train,
            ),
            batch_size=bs,
            num_workers=n_workers,
            pin_memory=str(device).startswith('cuda'),
        ),
        device=device,
    )
