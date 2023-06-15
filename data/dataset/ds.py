import torch

import ai

from aic.data.dataset.worker import TrainDataWorker


def train_data_iter(cfg, batch_size, device='cuda', n_workers=1, train=False):
    return ai.data.DataIterator(
        torch.utils.data.DataLoader(
            TrainDataWorker(cfg),
            batch_size=cfg.train.bs,
            num_workers=n_workers,
            pin_memory=device.startswith('cuda'),
        ),
        device=device,
    )
