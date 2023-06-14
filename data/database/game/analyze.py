from fire import Fire
import matplotlib.pyplot as plt

import cu

from aic.const import GAME_DB_PATH
from aic.data.database.game.db import GameDatabase


class CLI:
    def __init__(s, name='lichess/2023-01', train=False):
        path = GAME_DB_PATH / name
        if not path.exists():
            cu.storage.download_dir(path)

        s._db = GameDatabase(name, train)

    def run(s):
        s.overview()
        s.elos()

    def overview(s):
        print('n chunks:', s._db.n_chunks)

        chunk = s._db.load_chunk(0)
        print('n games per chunk:', len(chunk.games))
        print('n actions per chunk:', len(chunk.actions))

        n_games = 0
        n_actions = 0
        for chunk in s._db.chunk_iter():
            n_games += len(chunk.games)
            n_actions += len(chunk.actions)
        print('n games total:', n_games)
        print('n actions total:', n_actions)

    def elos(s, bins=32):
        elos = []
        elo_diffs = []
        for game in s._db.load_chunk(0).game_iter():
            elos.extend([game.meta.white_elo, game.meta.black_elo])
            elo_diffs.append(abs(game.meta.white_elo - game.meta.black_elo))

        plt.hist(elos, bins)
        plt.show()

        plt.hist(elo_diffs, bins)
        plt.show()


if __name__ == '__main__':
    Fire(CLI)
