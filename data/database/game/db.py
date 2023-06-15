from typing import Iterator
import random
import numpy as np

from aic.const import GAME_DB_PATH
from aic.data.database.game.game import CompressedGame
from aic.data.database.game.meta import LichessMeta


DB_TYPE_TO_META_CLS = {'lichess': LichessMeta}


class GameChunk:
    def __init__(s, games, actions, meta_cls):
        s.games = games
        s.actions = actions
        s._meta_cls = meta_cls

    @classmethod
    def load(cls, path, meta_cls):
        games = np.load(path / 'game.npy')
        actions = np.load(path / 'action.npy')
        return cls(games, actions, meta_cls)

    def __len__(s):
        return len(s.games)

    def game_iter(s, shuffle=False):
        idxs = range(len(s.games))
        if shuffle: idxs = random.shuffle(idxs)
        for i in idxs:
            yield s.get_game(i)

    def get_game(s, i):
        game = s.games[i]
        start, end = game[0], game[1]
        meta = s._meta_cls.from_list(game[2:])
        actions = s.actions[start:end]
        return CompressedGame(meta, actions)


class GameDatabase:
    def __init__(s, name: str, train: bool):
        path = GAME_DB_PATH / name / ('train' if train else 'val')
        assert path.exists()

        db_type = name.split('/')[0]
        s._meta_cls = DB_TYPE_TO_META_CLS[db_type]

        s._chunks = sorted(list(path.iterdir()), key=lambda x: int(x.stem))

    @property
    def n_chunks(s) -> int:
        return len(s._chunks)

    def chunk_iter(s, shuffle=False) -> Iterator[GameChunk]:
        idxs = range(s.n_chunks)
        if shuffle:
            idxs = list(idxs)
            random.shuffle(idxs)
        for i in idxs:
            yield s.load_chunk(i)

    def game_iter(s, shuffle=False) -> Iterator[CompressedGame]:
        for chunk in s.chunk_iter(shuffle):
            for game in chunk.game_iter(shuffle):
                yield game

    def load_chunk(s, i):
        return GameChunk.load(s._chunks[i], s._meta_cls)
