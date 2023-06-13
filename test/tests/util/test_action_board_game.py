import pytest

from aic.test import LICHESS_TEST_DATA_PATH
from aic.util import move_to_action, action_to_move, pcb_to_board, board_to_pcb, eq_boards, pgn_to_game
from aic.util.lichess import pgn_iterator


@pytest.fixture
def lichess_games():
    games = []

    with open(LICHESS_TEST_DATA_PATH / 'lichess_medium.pgn', 'r') as f:
        for pgn in pgn_iterator(f):
            games.append(pgn_to_game(pgn))

    with open(LICHESS_TEST_DATA_PATH / 'underpromo.pgn', 'r') as f:
        games.append(pgn_to_game(f.read()))

    return games


def test_move_conversion(lichess_games):
    for game in lichess_games():
        for state in game.mainline():
            if state.parent is None:
                continue
            move = state.move
            board = state.parent.board()
            player = 1 if state.parent.turn() else -1
            action = move_to_action(move, player)
            assert action_to_move(action, board) == move


def test_board_conversion(lichess_games):
    for game in lichess_games():
        for state in game.mainline():
            pcb = state.board()
            player = 1 if state.turn() else -1
            board = pcb_to_board(pcb)
            assert eq_boards(board_to_pcb(board, player), pcb)


def test_mirror():
    with open(LICHESS_TEST_DATA_PATH / 'mirror/a.pgn', 'r') as f:
        a = pgn_to_game(f.read())
    with open(LICHESS_TEST_DATA_PATH / 'mirror/b.pgn', 'r') as f:
        b = pgn_to_game(f.read())

    a = a.next().next().next()
    b = b.next().next().next().next()
    p1 = 1 if a.parent.turn() else -1
    p2 = 1 if b.parent.turn() else -1
    assert p1 == 1
    assert p2 == -1

    while a is not None and b is not None:
        action1 = move_to_action(a.move, p1)
        action2 = move_to_action(b.move, p2)
        assert action1 == action2

        assert (1 if a.parent.board().turn else -1) == p1
        board1 = pcb_to_board(a.parent.board())
        assert (1 if b.parent.board().turn else -1) == p2
        board2 = pcb_to_board(b.parent.board())
        assert (board1 == board2).all()

        a, b = a.next(), b.next()
        p1, p2 = -p1, -p2
    assert a is None and b is None
