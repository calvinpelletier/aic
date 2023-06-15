import torch

import ai

from aic.data.dataset.worker import TrainDataWorker


def train_data_iter(cfg, device='cuda', n_workers=1, include_ply=False):
    return ai.data.DataIterator(
        torch.utils.data.DataLoader(
            TrainDataWorker(cfg, include_ply),
            batch_size=cfg.train.bs,
            num_workers=n_workers,
            pin_memory=str(device).startswith('cuda'),
        ),
        device=device,
    )
