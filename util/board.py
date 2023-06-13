import chess
import numpy as np
from enum import Enum
from numpy.typing import NDArray


BOARD_W = 8
BOARD_H = 8
BOARD_AREA = BOARD_W * BOARD_H
BOARD_SHAPE = (BOARD_H, BOARD_W)

EP_RANKS = [2, 5]

# square state ~
EMPTY = 0
EMPTY_EP = 1
MY_PAWN = 2
MY_KNIGHT = 3
MY_BISHOP = 4
MY_ROOK = 5
MY_ROOK_CASTLE = 6
MY_QUEEN = 7
MY_KING = 8
OP_PAWN = 9
OP_KNIGHT = 10
OP_BISHOP = 11
OP_ROOK = 12
OP_ROOK_CASTLE = 13
OP_QUEEN = 14
OP_KING = 15

N_SQUARE_STATES = 16

PIECE_TO_SQUARE_STATE = {
    1: MY_PAWN,
    -1: OP_PAWN,

    2: MY_KNIGHT,
    -2: OP_KNIGHT,

    3: MY_BISHOP,
    -3: OP_BISHOP,

    4: MY_ROOK,
    -4: OP_ROOK,

    5: MY_QUEEN,
    -5: OP_QUEEN,

    6: MY_KING,
    -6: OP_KING,
}

SQUARE_STATE_TO_PIECE = {v: k for k, v in PIECE_TO_SQUARE_STATE.items()}
SQUARE_STATE_TO_PIECE[EMPTY] = None
SQUARE_STATE_TO_PIECE[EMPTY_EP] = None
SQUARE_STATE_TO_PIECE[MY_ROOK_CASTLE] = 4
SQUARE_STATE_TO_PIECE[OP_ROOK_CASTLE] = -4
# ~

# typing
Board = NDArray[np.uint8]


def pcb_to_board(pcb: chess.Board) -> Board:
    '''python-chess board to numpy board'''

    assert EMPTY == 0
    board = np.zeros(BOARD_SHAPE, dtype=np.uint8)

    player = cur_player(pcb)

    # pieces
    for square, piece in pcb.piece_map().items():
        side = color_to_side(piece.color, player)
        x, y = square_to_coords(square, player)
        board[y, x] = PIECE_TO_SQUARE_STATE[piece.piece_type * side]

    # castling rights
    for color in [True, False]:
        for castling_rights_fn, x in [(pcb.has_kingside_castling_rights, 7), (pcb.has_queenside_castling_rights, 0)]:
            if castling_rights_fn(color):
                y = _perspective(0, player) if color else _perspective(7, player)
                _set_castling_rights(board, x, y, color, player)

    # en passant
    if pcb.has_legal_en_passant():
        x, y = square_to_coords(pcb.ep_square, player)
        assert board[y, x] == EMPTY
        board[y, x] = EMPTY_EP

    return board


def board_to_pcb(board: Board, player: int) -> chess.Board:
    '''numpy board to python-chess board'''

    pcb = chess.Board()
    pcb.clear()
    pcb.turn = player == 1

    # pieces
    for y in range(BOARD_H):
        for x in range(BOARD_W):
            piece = SQUARE_STATE_TO_PIECE[board[y, x]]
            if piece is not None:
                pcb.set_piece_at(
                    coords_to_square(x, y, player),
                    chess.Piece(abs(piece), side_to_color(1 if piece > 0 else -1, player)),
                )

    # castling rights
    castle_fen = ''
    white_rook_castle = MY_ROOK_CASTLE if player == 1 else OP_ROOK_CASTLE
    black_rook_castle = MY_ROOK_CASTLE if player == -1 else OP_ROOK_CASTLE
    if board[_perspective(0, player), 7] == white_rook_castle:
        castle_fen += 'K'
    if board[_perspective(0, player), 0] == white_rook_castle:
        castle_fen += 'Q'
    if board[_perspective(7, player), 7] == black_rook_castle:
        castle_fen += 'k'
    if board[_perspective(7, player), 0] == black_rook_castle:
        castle_fen += 'q'
    pcb.set_castling_fen(castle_fen if castle_fen else '-')

    # en passant
    for y in EP_RANKS:
        for x in range(BOARD_W):
            if board[y, x] == EMPTY_EP:
                assert pcb.ep_square is None
                pcb.ep_square = coords_to_square(x, y, player)

    return pcb


def square_to_coords(square: chess.Square, player: int) -> tuple[int, int]:
    x = chess.square_file(square)
    y = _perspective(chess.square_rank(square), player)
    return x, y


def coords_to_square(x: int, y: int, player: int) -> chess.Square:
    return chess.square(x, _perspective(y, player))


def color_to_side(color: bool, player: int) -> int:
    return player if color else -player


def side_to_color(side: int, player: int) -> bool:
    return player == 1 if side == 1 else player == -1


def eq_boards(a: chess.Board, b: chess.Board) -> bool:
    '''check if boards are equal (same turn, FEN, en passant, and castling rights)'''

    if a.has_legal_en_passant() != b.has_legal_en_passant():
        return False

    if a.has_legal_en_passant() and a.ep_square != b.ep_square:
        return False

    return a.board_fen() == b.board_fen() and \
        a.castling_rights == b.castling_rights and \
        a.turn == b.turn


def pcb_to_key(pcb: chess.Board) -> str:
    fen = pcb.fen()
    pcb, turn, castling, ep, _, _ = fen.split(' ')
    return ' '.join([pcb, turn, castling, ep])


def cur_player(pcb: chess.Board) -> int:
    return 1 if pcb.turn else -1


def _set_castling_rights(board: Board, x: int, y: int, color: bool, player: int) -> None:
    if (color and player == 1) or (not color and player == -1):
        assert board[y, x] == MY_ROOK
        board[y, x] = MY_ROOK_CASTLE
    else:
        assert board[y, x] == OP_ROOK
        board[y, x] = OP_ROOK_CASTLE


def _perspective(y: int, player: int) -> int:
    if player == 1:
        return 7 - y
    elif player == -1:
        return y
    else:
        raise Exception(player)
