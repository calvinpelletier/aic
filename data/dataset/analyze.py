from fire import Fire
import matplotlib.pyplot as plt
from time import time

import cu

from aic.const import ELO_BINS
from aic.data.dataset.iterate import train_data_iter


class CLI:
    def __init__(s, task='bm2o', db='lichess/2023-01', bs=32):
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

    def elos(s, n_batches=10_000, d0=0., d1=0., d2=0., d3=0., d4=0., d5=0.):
        s._cfg.data.drop_chance_by_elo_bin = [d0, d1, d2, d3, d4]

        elos = []
        for i, batch in enumerate(train_data_iter(s._cfg)):
            for j in range(s._cfg.train.bs):
                elos.append(batch['meta'][j, 0])
            if i >= n_batches:
                break

        plt.hist(elos, [0] + ELO_BINS + [3500])
        plt.show()

    def speed(s, n_batches=1000, device='cuda', n_workers=1):
        start = time()
        for i, batch in enumerate(train_data_iter(s._cfg, device, n_workers)):
            if i >= n_batches:
                break
        end = time()
        seconds_per_batch = (end - start) / n_batches
        print('seconds per batch', seconds_per_batch)
        print('batches per second', 1. / seconds_per_batch)


if __name__ == '__main__':
    Fire(CLI)
