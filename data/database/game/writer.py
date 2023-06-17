import numpy as np

import cu

from aic.const import GAME_DB_PATH
from aic.data.database.game.game import CompressedGame


class GameDatabaseWriter:
    def __init__(s, name: str, chunk_size=100_000):
        path = GAME_DB_PATH / name
        if path.exists(): cu.disk.remove(path)
        s.path = path

        s.train = _GameDatabaseWriter(s.path / 'train', chunk_size)
        s.val = _GameDatabaseWriter(s.path / 'val', chunk_size)

    def flush(s):
        s.train.flush()
        s.val.flush()


class _GameDatabaseWriter:
    def __init__(s, path, chunk_size):
        s._path = path
        s._chunk_size = chunk_size
        s._cur_chunk_id = 0
        s._games = []
        s._actions = []

    def __del__(s):
        s.flush()

    def add(s, game: CompressedGame):
        s._add_game(game)

        if len(s._games) >= s._chunk_size:
            s.flush()

    def flush(s):
        if not len(s._games):
            return

        s._write_chunk()

        s._cur_chunk_id += 1
        s._games = []
        s._actions = []

    def _add_game(s, game):
        start = len(s._actions)
        s._actions.extend(game.actions)
        end = len(s._actions)
        s._games.append([start, end] + game.meta.as_list())

    def _write_chunk(s):
        path = s._path / str(s._cur_chunk_id)
        path.mkdir(parents=True)

        np.save(path / 'game.npy', np.asarray(s._games, dtype=np.int32))
        np.save(path / 'action.npy', np.asarray(s._actions, dtype=np.uint16))
