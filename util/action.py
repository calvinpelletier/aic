import chess
import numpy as np
from numpy.typing import NDArray

from aic.util.rel_action import RelActionMap
from aic.util.board import BOARD_AREA, BOARD_W, square_to_coords, coords_to_square, cur_player


REL_ACTION_MAP = RelActionMap()
REL_ACTION_SIZE = len(REL_ACTION_MAP)
ACTION_SIZE = BOARD_AREA * REL_ACTION_SIZE


def move_to_action(move: chess.Move, player: int) -> int:
    '''convert a chess move to an integer action from a player's perspective'''
    x1, y1, x2, y2 = move_to_coords(move, player)
    dx, dy = x2 - x1, y2 - y1
    rel_action = REL_ACTION_MAP.to_action(dx, dy, move.promotion)
    return relative_to_absolute_action(x1, y1, rel_action)


def action_to_move(action: int, board: chess.Board) -> chess.Move:
    '''convert an integer action to a chess move on a given board'''
    player = cur_player(board)
    x1, y1, rel_action = absolute_to_relative_action(action)
    dx, dy, underpromo = REL_ACTION_MAP.from_action(rel_action)
    x2, y2 = x1 + dx, y1 + dy
    return coords_to_move(board, player, x1, y1, x2, y2, underpromo)


def relative_to_absolute_action(x1: int, y1: int, rel_action: int) -> int:
    '''convert starting coords and a relative action into an absolute action'''
    return rel_action * BOARD_AREA + y1 * BOARD_W + x1


def absolute_to_relative_action(action: int) -> tuple[int, int, int]:
    '''convert an absolute action into starting coords and a relative action'''
    x1 = action % BOARD_W
    y1 = (action // BOARD_W) % BOARD_W
    rel_action = action // BOARD_AREA
    return x1, y1, rel_action


def move_to_coords(move: chess.Move, player: int) -> tuple[int, int, int, int]:
    x1, y1 = square_to_coords(move.from_square, player)
    x2, y2 = square_to_coords(move.to_square, player)
    return x1, y1, x2, y2


def coords_to_move(board: chess.Board, player: int, x1: int, y1: int, x2: int, y2: int, underpromo) -> chess.Move:
    square1 = coords_to_square(x1, y1, player)
    square2 = coords_to_square(x2, y2, player)
    return board.find_move(square1, square2, underpromo)


def legal_mask(board: chess.Board) -> NDArray[np.uint8]:
    '''return an array of length ACTION_SIZE where 0s and 1s indicate illegal and legal actions, respectively'''
    player = cur_player(board)
    ret = np.zeros(ACTION_SIZE, dtype=np.uint8)
    for move in board.legal_moves:
        ret[move_to_action(move, player)] = 1
    return ret
