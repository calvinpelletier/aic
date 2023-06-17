import torch
import numpy as np
import random
import chess
from typing import Optional

from aic.data.database import GameDatabase
from aic.task import build_task
from aic.util import action_to_move, elo_to_bin


class DataWorker(torch.utils.data.IterableDataset):
    def __init__(s,
        db_name: str,
        task_name: str,
        buf_size: int = 1,
        drop_chance_by_elo_bin: Optional[list[float]] = None,
        rand_trim_start: Optional[tuple[int, int]] = None,
        rand_trim_end: Optional[tuple[int, int]] = None,
        include_ply: bool = False,
        train: bool = False,
    ):
        super().__init__()
        s._drop_chance_by_elo_bin = drop_chance_by_elo_bin
        s._rand_trim_start = rand_trim_start
        s._rand_trim_end = rand_trim_end
        s._include_ply = include_ply
        s._train = train

        s._chunk_iter = _inf_chunk_iter(GameDatabase(db_name, train), shuffle=train)

        s._preprocessor = build_task(task_name).preprocess

        s._next_chunk()
        s._buf = [s._next_game() for _ in range(buf_size)]
        s._j = 0

    def __iter__(s):
        return s

    def __next__(s):
        data = s._preprocessor(s._buf[s._j])
        if s._include_ply:
            data['ply'] = s._buf[s._j].i

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

        return _Game(compressed.meta, compressed.actions, s._rand_trim_start, s._rand_trim_end)

    def _next_chunk(s):
        s._chunk = next(s._chunk_iter)
        s._idxs = np.random.permutation(len(s._chunk)) if s._train else list(range(len(s._chunk)))
        s._i = 0

    def _should_drop(s, game):
        if s._drop_chance_by_elo_bin is None:
            return False

        bin = elo_to_bin(max(game.meta.white_elo, game.meta.black_elo))
        return random.random() < s._drop_chance_by_elo_bin[bin]

def _inf_chunk_iter(db, shuffle):
    while 1:
        for chunk in db.chunk_iter(shuffle):
            yield chunk


class TaskDataWorker(torch.utils.data.IterableDataset):
    def __init__(s, task_name):
        s._task_name = task_name

    def __iter__(s):
        return _task_data_iter(s._task_name)

def _task_data_iter(task_name):
    preprocessor = build_task(task_name).preprocess
    chunk = GameChunk(
        np.load(TASK_DS_PATH / 'game.npy'),
        np.load(TASK_DS_PATH / 'action.npy'),
        LichessMeta,
    )
    for compressed in chunk.game_iter():
        game = _Game(compressed.meta, compressed.actions, None, None)
        done = False
        while not done:
            data = preprocessor(game)
            data['ply'] = game.i
            yield data
            done = game.next_position()
