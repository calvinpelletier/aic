import torch
import numpy as np
import random
import chess

from aic.data.database import GameDatabase
from aic.task import build_task
from aic.util import action_to_move, elo_to_bin


class TrainDataWorker(torch.utils.data.IterableDataset):
    def __init__(s, cfg):
        super().__init__()
        s._cfg = cfg
        s._db = GameDatabase(cfg.data.db, train=True)
        s._chunk_iter = _inf_chunk_iter(s._db)
        s._preprocessor = build_task(cfg).preprocess
        s._next_chunk()
        s._buf = [s._next_game() for _ in range(cfg.train.bs)]
        s._j = 0

    def __iter__(s):
        return s

    def __next__(s):
        data = s._preprocessor(s._buf[s._j])
        is_done = s._buf[s._j].next_position()
        if is_done:
            s._buf[s._j] = s._next_game()
        s._j = (s._j + 1) % len(s._buf)
        return data

    def _next_game(s):
        while 1:
            compressed = s._chunk.get_game(s._idxs[s._i])

            s._i += 1
            if s._i == len(s._chunk):
                s._next_chunk()

            if not s._should_drop(compressed):
                break

        return _Game(s._cfg.data, compressed.meta, compressed.actions)

    def _next_chunk(s):
        s._chunk = next(s._chunk_iter)
        s._idxs = np.random.permutation(len(s._chunk))
        s._i = 0

    def _should_drop(s, game):
        if s._cfg.data.drop_chance_by_elo_bin is None:
            return False

        bin = elo_to_bin(max(game.meta.white_elo, game.meta.black_elo))
        return random.random() < s._cfg.data.drop_chance_by_elo_bin[bin]

def _inf_chunk_iter(db):
    while 1:
        for chunk in db.chunk_iter(shuffle=True):
            yield chunk


class _Game:
    def __init__(s, cfg, meta, actions):
        s.outcome = meta.outcome
        s.white_elo = meta.white_elo
        s.black_elo = meta.black_elo
        s.tc_base = meta.tc_base
        s.tc_inc = meta.tc_inc

        s.actions = actions

        s.pcb = chess.Board()

        s._end = len(s.actions)
        if cfg.rand_trim_end is not None:
            x, y = cfg.rand_trim_end
            y = min(y, len(s.actions))
            if y > x:
                s._end = random.randrange(x, y)

        s.i = 0
        if cfg.rand_trim_start is not None:
            x, y = cfg.rand_trim_start
            y = min(y, len(s.actions) - 1)
            if y > x:
                for _ in range(random.randrange(x, y)):
                    done = s.next_position()
                    assert not done
                print(s.i, y)
                assert s.i == y

    def next_position(s):
        s.pcb.push(action_to_move(s.actions[s.i], s.pcb))
        s.i += 1
        return s.i == s._end
