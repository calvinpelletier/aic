import cu

from aic.const import DATA_PATH
from aic.data import GameDatabase
from aic.data.internalize.lichess.games import _pgns_to_game_db, _compress_game, _game_filter
from aic.test import LICHESS_TEST_DATA_PATH
from aic.util import pgn_to_game, actions_to_game, eq_games
from aic.util.lichess import pgn_iterator


def test_lichess_game_db():
    reader = cu.shell.get_stdout_reader(['zstdcat', str(LICHESS_TEST_DATA_PATH / 'lichess_medium.pgn.zst')])
    db_path, counts = _pgns_to_game_db(reader, 'lichess/test', 4)

    assert db_path == DATA_PATH / 'db/game/lichess/test'
    assert counts == {
        'min game len': 5,
        'time control': 19,
        'valid': 26,
        'total': 50,
    }

    train_db = GameDatabase('lichess/test', train=True)
    val_db = GameDatabase('lichess/test', train=False)

    assert train_db.n_chunks == 7
    assert val_db.n_chunks == 1
    for i, chunk in enumerate(train_db.chunk_iter()):
        expected_length = 4 if i < 6 else 1
        assert len(chunk) == expected_length
    assert len(val_db.load_chunk(0)) == 1

    games = list(val_db.game_iter()) + list(train_db.game_iter())
    gt_games = list(_lichess_games())
    assert 26 == len(games) == len(gt_games)
    for a, b in zip(games, gt_games):
        assert a.meta == _compress_game(b).meta
        if not a.meta.truncated:
            assert eq_games(actions_to_game(a.actions), b)


def _lichess_games():
    with open(LICHESS_TEST_DATA_PATH / 'lichess_medium.pgn', 'r') as f:
        for pgn in pgn_iterator(f):
            game = pgn_to_game(pgn)
            if not _game_filter(game):
                yield game
