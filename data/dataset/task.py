from fire import Fire
from collections import defaultdict
import numpy as np

from aic.const import TASK_DS_PATH
from aic.data.database.game.db import GameDatabase
from aic.util import elo_to_bin, is_underpromo


N_GAMES_PER_ELO_BIN = 256


class CLI:
    def create(s):
        # read and organize ~
        db = GameDatabase('lichess/2023-01', train=False)
        games_by_elo_bin = defaultdict(list)
        underpromo_games = []
        for game in db.game_iter():
            # games by elo bin
            elo = max([game.meta.white_elo, game.meta.black_elo])
            bin = elo_to_bin(elo)
            if bin > 0 and len(games_by_elo_bin[bin]) < N_GAMES_PER_ELO_BIN:
                games_by_elo_bin[bin].append(game)

            # underpromo games
            if _game_has_underpromo(game):
                underpromo_games.append(game)
        # ~

        # write
        for bin, games in games_by_elo_bin.items():
            _write_games(games, TASK_DS_PATH / f'elo{bin}')
        _write_games(underpromo_games, TASK_DS_PATH / 'underpromo')

    def examine(s):
        # elo
        n_total_actions = 0
        for dir in TASK_DS_PATH.iterdir():
            if dir.name.startswith('elo'):
                print(dir.name, 'n games', len(np.load(dir / 'game.npy')))
                n_actions = len(np.load(dir / 'action.npy'))
                print(dir.name, 'n actions', n_actions)
                n_total_actions += n_actions
        print('n total actions', n_total_actions)

        # underpromo
        print('underpromo n games', len(np.load(TASK_DS_PATH / 'underpromo/game.npy')))


def _write_games(games, path):
    # combine
    game_data = []
    actions = []
    for game in games:
        start = len(actions)
        actions.extend(game.actions)
        end = len(actions)
        game_data.append([start, end] + game.meta.as_list())

    # write
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / 'game.npy', np.asarray(game_data, dtype=np.int32))
    np.save(path / 'action.npy', np.asarray(actions, dtype=np.uint16))


def _game_has_underpromo(game):
    for action in game.actions:
        if is_underpromo(action):
            return True
    return False


if __name__ == '__main__':
    Fire(CLI)
