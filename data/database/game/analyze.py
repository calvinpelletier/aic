from fire import Fire
import matplotlib.pyplot as plt
from collections import defaultdict

import cu

from aic.const import GAME_DB_PATH, ELO_BINS
from aic.data.database.game.db import GameDatabase
from aic.util import elo_to_bin, is_underpromo


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

    def elo(s, both=False):
        elos = []
        elo_diffs = []
        for game in s._db.load_chunk(0).game_iter():
            elo = [game.meta.white_elo, game.meta.black_elo]
            elos.extend(elo) if both else elos.append(max(elo))
            elo_diffs.append(abs(game.meta.white_elo - game.meta.black_elo))

        counts_by_bin = defaultdict(int)
        for elo in elos:
            counts_by_bin[elo_to_bin(elo)] += 1
        for i in range(len(ELO_BINS) + 1):
            print(i, counts_by_bin[i])

        plt.hist(elos, [0] + ELO_BINS + [3500])
        plt.show()

        plt.hist(elo_diffs, 16)
        plt.show()

    def underpromo(s):
        chunk = s._db.load_chunk(0)
        print('n games', len(chunk))

        n_games_with_underpromos = 0
        for game in chunk.game_iter():
            for action in game.actions:
                if is_underpromo(action):
                    n_games_with_underpromos += 1
                    break
        print('n games with underpromos', n_games_with_underpromos)


if __name__ == '__main__':
    Fire(CLI)
