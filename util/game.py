from chess.pgn import read_game, Game
import io
from typing import Iterable

from aic.util.action import action_to_move


def pgn_to_game(pgn: str) -> Game:
    game = read_game(io.StringIO(pgn))
    if len(game.errors):
        raise Exception(str(game.errors))
    return game


def actions_to_game(actions: Iterable[int]) -> Game:
    game = Game()
    node = game
    for action in actions:
        move = action_to_move(action, node.board())
        node = node.add_variation(move)
    return game


def eq_games(a: Game, b: Game) -> bool:
    '''check if games are equal'''
    
    a = a.next()
    b = b.next()
    while a is not None and b is not None:
        if a.move != b.move:
            return False
        a, b = a.next(), b.next()
    return a is None and b is None
