import numpy as np

from aic.util import pcb_to_board, legal_mask


def board(data, game):
    data['board'] = pcb_to_board(game.pcb)


def outcome(data, game):
    data['outcome'] = np.int8(game.outcome * (1 if game.i % 2 == 0 else -1))


def action(data, game):
    data['action'] = game.actions[game.i].astype(np.int16)


def legal(data, game):
    data['legal'] = legal_mask(game.pcb)


def history(data, game, max_length=127):
    assert game.i <= max_length
    data['history_len'] = np.uint8(game.i)
    data['history'] = np.zeros(max_length, dtype=np.int16)
    if game.i > 0: data['history'][:game.i] = game.actions[:game.i]


def meta(data, game):
    data['meta'] = np.array([
        game.white_elo if game.i % 2 == 0 else game.black_elo,
        game.tc_base,
        game.tc_inc,
    ], dtype=np.int32)
