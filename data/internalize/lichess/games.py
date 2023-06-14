from collections import defaultdict
from fire import Fire

import cu

from aic.data.database.game import CompressedGame, GameDatabaseWriter, LichessMeta
from aic.util import move_to_action, pgn_to_game
from aic.util.lichess import pgn_iterator


URL = 'https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst'
DOWNLOAD_PATH = '/tmp/lichess_db_standard_rated_{}.pgn.zst'
DB_NAME = 'lichess/{}'

MIN_TIME_CONTROL = 180
MAX_TIME_CONTROL = 60 * 30
MIN_GAME_LEN = 5
MAX_GAME_LEN = 128
TRUNCATE = True

RESULT_ENC = {
    '1-0': 1,
    '0-1': -1,
    '1/2-1/2': 0,
}

TERMINATION_ENC = {
    'Normal': 0,
    'Abandoned': 1,
    'Time forfeit': 2,
    'Rules infraction': None,
}
ALL_TERMINATIONS = set(TERMINATION_ENC.keys())
VALID_TERMINATIONS = set([termination for termination, enc in TERMINATION_ENC.items() if enc is not None])


class CLI:
    def __init__(s, month='2023-01'):
        s._month = month

    def internalize(s):
        s.download()
        s.convert()
        s.clean_up()

    def download(s):
        cu.web.download_file_from_url(URL.format(s._month), DOWNLOAD_PATH.format(s._month))

    def convert(s):
        reader = cu.shell.get_stdout_reader(['zstdcat', DOWNLOAD_PATH.format(s._month)])
        db_path, counts = _pgns_to_game_db(reader, DB_NAME.format(s._month))
        cu.storage.upload_dir(db_path)
        for k, v in counts.items(): print(k, v)

    def clean_up(s):
        cu.disk.remove(DOWNLOAD_PATH.format(s._month))


def _pgns_to_game_db(reader, db_name, chunk_size=100_000, train_val_ratio=1000):
    writer = GameDatabaseWriter(db_name, chunk_size)
    counter = defaultdict(int)

    for pgn in pgn_iterator(reader):
        counter['total'] += 1

        # convert pgn to game
        try:
            game = pgn_to_game(pgn)
        except Exception as e:
            counter[str(e)] += 1
            continue

        # check if we should skip
        filter_reason = _game_filter(game)
        if filter_reason:
            counter[filter_reason] += 1
            continue

        # add compressed game to database
        compressed = _compress_game(game)
        if counter['valid'] % train_val_ratio == 0:
            writer.val.add(compressed)
        else:
            writer.train.add(compressed)

        counter['valid'] += 1

    writer.flush()
    return writer.path, counter


def _compress_game(game):
    max_len = MAX_GAME_LEN if TRUNCATE else None

    outcome = RESULT_ENC[game.headers['Result']]
    white_elo, black_elo = map(
        lambda x: int(game.headers[x]),
        ['WhiteElo', 'BlackElo'],
    )
    tc_base, tc_inc = map(
        lambda x: int(x),
        game.headers['TimeControl'].split('+'),
    )
    termination = TERMINATION_ENC[game.headers['Termination']]

    actions = []
    truncated = False
    for state in game.mainline():
        if state.move is None:
            continue

        if max_len is not None and state.ply() > max_len:
            truncated = True
            break

        action = move_to_action(state.move, 1 if state.parent.turn() else -1)
        actions.append(action)

    meta = LichessMeta(outcome, white_elo, black_elo, tc_base, tc_inc, termination, int(truncated))
    return CompressedGame(meta, actions)


def _game_filter(game):
    # no result
    if game.headers['Result'] == '*':
        return 'no result'

    # invalid termination
    termination = game.headers['Termination']
    # assert termination in ALL_TERMINATIONS
    if termination not in VALID_TERMINATIONS:
        if termination not in ALL_TERMINATIONS:
            print(termination)
            raise Exception(termination)
        return 'termination'

    # time control
    tc = game.headers['TimeControl']
    if tc == '-': # correspondence
        return 'time control'
    base, increment = map(lambda x: int(x), tc.split('+'))
    if base < MIN_TIME_CONTROL or base > MAX_TIME_CONTROL:
        return 'time control'

    # game len
    game_len = game.end().ply()
    if game_len < MIN_GAME_LEN:
        return 'min game len'
    if not TRUNCATE and game_len > MAX_GAME_LEN:
        return 'max game len'

    return None


if __name__ == '__main__':
    Fire(CLI)
