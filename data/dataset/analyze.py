from fire import Fire
import matplotlib.pyplot as plt
from time import time

import cu

from aic.const import ELO_BINS, PLY_BINS
from aic.data.dataset.iterate import train_data_iter


class CLI:
    def __init__(s, task='bm2o', db='lichess/2023-01', bs=32, device='cuda', n_workers=6, n_batches=10_000):
        s._cfg = cu.Config({
            'task': task,
            'data': {
                'db': db,
                'rand_trim_start': (0, 16),
                'rand_trim_end': (64, 128),
                'drop_chance_by_elo_bin': [1., .9, .8, .7, .6, 0.],
                'include_ply': False,
            },
            'train': {'bs': bs},
        })
        s._device = device
        s._n_workers = n_workers
        s._n_batches = n_batches

    def elo(s):
        elos_drop = s._get_elos()

        s._cfg.data.drop_chance_by_elo_bin = None
        elos_reg = s._get_elos()

        bins = [0] + ELO_BINS + [3500]
        plt.hist(elos_reg, bins, alpha=0.5, label='reg')
        plt.hist(elos_drop, bins, alpha=0.5, label='drop')
        plt.legend(loc='upper right')
        plt.show()

    def speed(s, max_workers=10):
        speeds = []
        for n_workers in range(max_workers):
            print('n workers', n_workers)
            start = time()
            for i, batch in enumerate(train_data_iter(s._cfg, s._device, n_workers)):
                if i >= s._n_batches:
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

    def ply(s):
        plys_trim = s._get_plys()

        s._cfg.data.rand_trim_start = None
        s._cfg.data.rand_trim_end = None
        plys_reg = s._get_plys()

        bins = [0] + PLY_BINS + [128]
        plt.hist(plys_reg, bins, alpha=0.5, label='reg')
        plt.hist(plys_trim, bins, alpha=0.5, label='trim')
        plt.legend(loc='upper right')
        plt.show()

    def _get_elos(s):
        elos = []
        for i, batch in enumerate(train_data_iter(s._cfg, s._device, s._n_workers)):
            for j in range(s._cfg.train.bs):
                elos.append(batch['meta'][j, 0].item())
            if i >= s._n_batches:
                break
        return elos

    def _get_plys(s):
        plys = []
        for i, batch in enumerate(train_data_iter(s._cfg, s._device, s._n_workers, include_ply=True)):
            for j in range(s._cfg.train.bs):
                plys.append(batch['ply'][j].item())
            if i >= s._n_batches:
                break
        return plys


if __name__ == '__main__':
    Fire(CLI)
