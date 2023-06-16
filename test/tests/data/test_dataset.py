import torch

import cu

from ai.util import assert_shape, assert_bounds

from aic.const import ELO_BINS, PLY_BINS
from aic.data.dataset.iterate import data_iters, train_data_iter
from aic.data.dataset.util import measure_data_speed
from aic.test.const import DEVICE
from aic.util.action import ACTION_SIZE


BS = 32


def test_bmh2oa():
    train_data, val_data = _data_iters('bmh2oa', 2048, 256)

    _check_bmh2oa_batch(next(iter(train_data)), BS)

    assert len(val_data) == 8
    _check_bmh2oa_batch(next(iter(val_data)), 256)


def test_b2l():
    batch = next(iter(_train_data_iter('b2l')))
    assert len(batch) == 2

    x = batch['legal']
    assert x.device == DEVICE
    assert x.dtype == torch.uint8
    assert_shape(x, [BS, ACTION_SIZE])
    assert_bounds(x, [0, 1])


def test_speed():
    batches_per_second = measure_data_speed(_train_data_iter('bmh2oa'))
    assert 300 < batches_per_second < 400


def test_trim():
    n_early = _count_early_ply(False)
    assert 700 < n_early < 1000
    n_early = _count_early_ply(True)
    assert 200 < n_early < 400


def test_drop():
    n_low = _count_low_elo(False)
    assert 1600 < n_low < 3000
    n_low = _count_low_elo(True)
    assert 10 < n_low < 400


def _check_bmh2oa_batch(batch, bs):
    assert len(batch) == 6

    x = batch['board']
    assert x.device == DEVICE
    assert x.dtype == torch.uint8
    assert_shape(x, [bs, 8, 8])
    assert_bounds(x, [0, 15])

    x = batch['meta']
    assert x.device == DEVICE
    assert x.dtype == torch.int32
    assert_shape(x, [bs, 3])
    assert_bounds(x, [0, 3500])

    x = batch['history']
    assert x.device == DEVICE
    assert x.dtype == torch.int16
    assert_shape(x, [bs, 127])
    assert_bounds(x, [0, ACTION_SIZE - 1])

    x = batch['history_len']
    assert x.device == DEVICE
    assert x.dtype == torch.uint8
    assert_shape(x, [bs])
    assert_bounds(x, [0, 127])

    x = batch['outcome']
    assert x.device == DEVICE
    assert x.dtype == torch.int8
    assert_shape(x, [bs])
    assert_bounds(x, [-1, 1])

    x = batch['action']
    assert x.device == DEVICE
    assert x.dtype == torch.int16
    assert_shape(x, [bs])
    assert_bounds(x, [0, ACTION_SIZE - 1])


def _count_early_ply(trim):
    n_early = 0
    for i, batch in enumerate(_train_data_iter('b2a', trim=trim, include_ply=True)):
        if i >= 200:
            break

        for j in range(BS):
            ply = batch['ply'][j].item()
            if ply < PLY_BINS[0]:
                n_early += 1
    return n_early


def _count_low_elo(drop):
    n_low = 0
    for i, batch in enumerate(_train_data_iter('bm2o', drop=drop)):
        if i >= 200:
            break

        for j in range(BS):
            elo = batch['meta'][j, 0].item()
            if elo < ELO_BINS[0]:
                n_low += 1
    return n_low


def _data_iters(task, n_val_samples, val_bs, trim=True, drop=True, include_ply=False):
    return data_iters(_cfg(task, trim, drop), n_val_samples, val_bs, DEVICE, include_ply, 0)


def _train_data_iter(task, trim=True, drop=True, include_ply=False):
    return train_data_iter(_cfg(task, trim, drop), DEVICE, include_ply, 0)


def _cfg(task, trim, drop):
    return cu.Config({
        'task': task,
        'data': {
            'db': 'lichess/2023-01',
            'rand_trim_start': (0, 16) if trim else None,
            'rand_trim_end': (64, 128) if trim else None,
            'drop_chance_by_elo_bin': [1., .9, .8, .7, .6, 0.] if drop else None,
        },
        'train': {'bs': BS},
    })
