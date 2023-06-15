from fire import Fire
import matplotlib.pyplot as plt
from time import time

import cu

from aic.const import ELO_BINS
from aic.data.dataset.iterate import train_data_iter


class CLI:
    def __init__(s, task='bm2o', db='lichess/2023-01', bs=32, device='cuda'):
        s._cfg = cu.Config({
            'task': task,
            'data': {
                'db': db,
                'rand_trim_start': (0, 16),
                'rand_trim_end': (64, 128),
                'drop_chance_by_elo_bin': None,
            },
            'train': {'bs': bs},
        })
        s._device = device

    def elo(s, n_batches=10_000, n_workers=6, d0=1., d1=.9, d2=.8, d3=.7, d4=.6, d5=0.):
        s._cfg.data.drop_chance_by_elo_bin = [d0, d1, d2, d3, d4, d5]

        elos = []
        for i, batch in enumerate(train_data_iter(s._cfg, s._device, n_workers)):
            for j in range(s._cfg.train.bs):
                elos.append(batch['meta'][j, 0].item())
            if i >= n_batches:
                break

        plt.hist(elos, [0] + ELO_BINS + [3500])
        plt.show()

    def speed(s, n_batches=1000, max_workers=10):
        speeds = []
        for n_workers in range(max_workers):
            print('n workers', n_workers)
            start = time()
            for i, batch in enumerate(train_data_iter(s._cfg, s._device, n_workers)):
                if i >= n_batches:
                    break
            end = time()
            seconds_per_batch = (end - start) / n_batches
            print('seconds per batch', seconds_per_batch)
            batches_per_second = 1. / seconds_per_batch
            print('batches per second', batches_per_second)
            speeds.append(batches_per_second)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        plt.plot(speeds)
        plt.show()


if __name__ == '__main__':
    Fire(CLI)
