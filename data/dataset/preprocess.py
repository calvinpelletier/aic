import numpy as np

from aic.util import pcb_to_board, legal_mask


def board(game):
    return pcb_to_board(game.pcb)


def outcome(game):
    return np.int8(game.outcome * (1 if game.i % 2 == 0 else -1))


def action(game):
    return game.actions[game.i].astype(np.int16)


def legal(game):
    return legal_mask(game.pcb)


def meta(game):
    return np.array([
        game.white_elo if game.i % 2 == 0 else game.black_elo,
        game.tc_base,
        game.tc_inc,
    ], dtype=np.int32)


def history(game, max_length=127):
    assert game.i <= max_length
    history_len = np.uint8(game.i)
    history = np.zeros(max_length, dtype=np.int16)
    if game.i > 0: history[:game.i] = game.actions[:game.i]
    return history, history_len


def ply(game):
    return np.uint8(game.i)


def full(game):
    data = {}
    data['board'] = board(game)
    data['outcome'] = outcome(game)
    data['action'] = action(game)
    data['legal'] = legal(game)
    data['meta'] = meta(game)
    data['history'], data['history_len'] = history(game)
    data['ply'] = ply(game)
    return data
