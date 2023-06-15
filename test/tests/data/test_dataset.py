import cu

from aic.data.dataset.iterate import train_data_iter
from aic.test.const import DEVICE


def test_b2l():
    batch = next(iter(train_data_iter(_cfg('b2l'), DEVICE, 0)))
    assert len(batch) == 2


def _cfg(task):
    return cu.Config({
        'task': task,
        'data': {
            'db': 'lichess/2023-01',
            'rand_trim_start': (0, 16),
            'rand_trim_end': (64, 128),
            'drop_chance_by_elo_bin': [1., .9, .8, .7, .6, 0.],
        },
        'train': {'bs': 32},
    })
