import torch

import ai

from aic.data.dataset.worker import TrainDataWorker


def train_data_iter(cfg, device='cuda', n_workers=1):
    return ai.data.DataIterator(
        torch.utils.data.DataLoader(
            TrainDataWorker(cfg),
            batch_size=cfg.train.bs,
            num_workers=n_workers,
            pin_memory=device.startswith('cuda'),
        ),
        device=device,
    )