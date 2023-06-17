from fire import Fire
from collections import defaultdict
import numpy as np

import cu

from aic.const import TASK_DS_PATH, UNDERPROMO_DS_PATH
from aic.data.database.game.db import GameDatabase
from aic.data.dataset import preprocess
from aic.data.dataset.game import Game
from aic.util import elo_to_bin, is_underpromo


N_TASK_GAMES_PER_ELO_BIN = 256


class CLI:
    def task(s):
        '''create an elo-balanced dataset of chess games for use in tasks'''

        writer = _DatasetWriter()

        db = GameDatabase('lichess/2023-01', train=False)
        count_by_elo_bin = defaultdict(int)
        for game in db.game_iter():
            elo = max([game.meta.white_elo, game.meta.black_elo])
            bin = elo_to_bin(elo)
            if bin > 0 and count_by_elo_bin[bin] < N_TASK_GAMES_PER_ELO_BIN:
                writer.add(game)
                count_by_elo_bin[bin] += 1

        if TASK_DS_PATH.exists():
            cu.disk.remove(TASK_DS_PATH)
        writer.write(TASK_DS_PATH)

    def underpromo(s):
        writer = _DatasetWriter()

        db = GameDatabase('lichess/2023-01', train=False)
        for game in db.game_iter():
            if _game_has_underpromo(game):
                writer.add(game, only_underpromos=True)

        if UNDERPROMO_DS_PATH.exists():
            cu.disk.remove(UNDERPROMO_DS_PATH)
        writer.write(UNDERPROMO_DS_PATH)


class _DatasetWriter:
    def __init__(s):
        s._data = defaultdict(list)

    def __len__(s):
        return len(s._data['action'])

    def add(s, compressed, only_underpromos=False):
        game = Game(compressed.meta, compressed.actions)
        done = False
        while not done:
            if not only_underpromos or is_underpromo(game.actions[game.i]):
                for k, v in preprocess.full(game).items():
                    s._data[k].append(v)
            done = game.next_position()

    def write(s, path):
        path.mkdir(parents=True, exist_ok=True)
        for k, data in s._data.items():
            np.save(path / f'{k}.npy', np.stack(data))


def _game_has_underpromo(game):
    for action in game.actions:
        if is_underpromo(action):
            return True
    return False


if __name__ == '__main__':
    Fire(CLI)
