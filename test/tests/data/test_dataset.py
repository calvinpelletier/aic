import torch

import cu

from ai.util import assert_shape, assert_bounds

from aic.data.dataset.iterate import train_data_iter
from aic.data.dataset.util import measure_data_speed
from aic.test.const import DEVICE
from aic.util.action import ACTION_SIZE


def test_bmh2oa():
    batch = next(iter(_data_iter('bmh2oa')))
    assert len(batch) == 6

    x = batch['board']
    assert x.device == DEVICE
    assert x.dtype == torch.uint8
    assert_shape(x, [32, 8, 8])
    assert_bounds(x, [0, 15])

    x = batch['meta']
    assert x.device == DEVICE
    assert x.dtype == torch.int32
    assert_shape(x, [32, 3])
    assert_bounds(x, [0, 3500])

    x = batch['history']
    assert x.device == DEVICE
    assert x.dtype == torch.int16
    assert_shape(x, [32, 127])
    assert_bounds(x, [0, ACTION_SIZE - 1])

    x = batch['history_len']
    assert x.device == DEVICE
    assert x.dtype == torch.uint8
    assert_shape(x, [32])
    assert_bounds(x, [0, 127])

    x = batch['outcome']
    assert x.device == DEVICE
    assert x.dtype == torch.int8
    assert_shape(x, [32])
    assert_bounds(x, [-1, 1])

    x = batch['action']
    assert x.device == DEVICE
    assert x.dtype == torch.int16
    assert_shape(x, [32])
    assert_bounds(x, [0, ACTION_SIZE - 1])


def test_b2l():
    batch = next(iter(_data_iter('b2l')))
    assert len(batch) == 2

    x = batch['legal']
    assert x.device == DEVICE
    assert x.dtype == torch.uint8
    assert_shape(x, [32, ACTION_SIZE])
    assert_bounds(x, [0, 1])


def test_speed():
    batches_per_second = measure_data_speed(_data_iter('bmh2oa'))
    assert 300 < batches_per_second < 400


def _data_iter(task):
    return train_data_iter(_cfg(task), DEVICE, 0)


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
