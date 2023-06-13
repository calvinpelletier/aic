import chess
import numpy as np
from enum import Enum


BOARD_W = 8
BOARD_H = 8
BOARD_AREA = BOARD_W * BOARD_H
BOARD_SHAPE = (BOARD_H, BOARD_W)

EP_RANKS = [2, 5]

# square state ~
EMPTY = 0
EMPTY_EP = 1
WHITE_PAWN = 2
WHITE_KNIGHT = 3
WHITE_BISHOP = 4
WHITE_ROOK = 5
WHITE_ROOK_CASTLE = 6
WHITE_QUEEN = 7
WHITE_KING = 8
BLACK_PAWN = 9
BLACK_KNIGHT = 10
BLACK_BISHOP = 11
BLACK_ROOK = 12
BLACK_ROOK_CASTLE = 13
BLACK_QUEEN = 14
BLACK_KING = 15

N_SQUARE_STATES = 16

PIECE_TO_SQUARE_STATE = {
    1: WHITE_PAWN,
    -1: BLACK_PAWN,

    2: WHITE_KNIGHT,
    -2: BLACK_KNIGHT,

    3: WHITE_BISHOP,
    -3: BLACK_BISHOP,

    4: WHITE_ROOK,
    -4: BLACK_ROOK,

    5: WHITE_QUEEN,
    -5: BLACK_QUEEN,

    6: WHITE_KING,
    -6: BLACK_KING,
}

SQUARE_STATE_TO_PIECE = {v: k for k, v in PIECE_TO_SQUARE_STATE.items()}
SQUARE_STATE_TO_PIECE[EMPTY] = None
SQUARE_STATE_TO_PIECE[EMPTY_EP] = None
SQUARE_STATE_TO_PIECE[WHITE_ROOK_CASTLE] = 4
SQUARE_STATE_TO_PIECE[BLACK_ROOK_CASTLE] = -4
# ~


def pcb_to_board(pcb):
    player = cur_player(pcb)

    assert EMPTY == 0
    board = np.zeros(BOARD_SHAPE, dtype=np.uint8)

    for square, piece in pcb.piece_map().items():
        side = color_to_side(piece.color, player)
        x, y = square_to_coords(square, player)
        board[y, x] = PIECE_TO_SQUARE_STATE[piece.piece_type * side]

    for color in [True, False]:
        for castling_rights_fn, x in [(pcb.has_kingside_castling_rights, 7), (pcb.has_queenside_castling_rights, 0)]:
            if castling_rights_fn(color):
                y = _perspective(0, player) if color else _perspective(7, player)
                _set_castling_rights(board, x, y, color)

    if pcb.has_legal_en_passant():
        x, y = square_to_coords(pcb.ep_square, player)
        assert board[y, x] == EMPTY
        board[y, x] = EMPTY_EP

    return board


def board_to_pcb(board, player):
    pcb = chess.Board()
    pcb.clear()
    pcb.turn = player == 1

    for y in range(BOARD_H):
        for x in range(BOARD_W):
            piece = SQUARE_STATE_TO_PIECE[board[y, x]]
            if piece is not None:
                pcb.set_piece_at(
                    coords_to_square(x, y, player),
                    chess.Piece(abs(piece), side_to_color(1 if piece > 0 else -1, player)),
                )

    castle_fen = ''
    if board[_perspective(0, player), 7] == WHITE_ROOK_CASTLE:
        castle_fen += 'K'
    if board[_perspective(0, player), 0] == WHITE_ROOK_CASTLE:
        castle_fen += 'Q'
    if board[_perspective(7, player), 7] == BLACK_ROOK_CASTLE:
        castle_fen += 'k'
    if board[_perspective(7, player), 0] == BLACK_ROOK_CASTLE:
        castle_fen += 'q'
    pcb.set_castling_fen(castle_fen if castle_fen else '-')

    for y in EP_RANKS:
        for x in range(BOARD_W):
            if board[y, x] == EMPTY_EP:
                assert pcb.ep_square is None
                pcb.ep_square = coords_to_square(x, y, player)

    return pcb


def square_to_coords(square, player):
    x = chess.square_file(square)
    y = _perspective(chess.square_rank(square), player)
    return x, y


def coords_to_square(x, y, player):
    return chess.square(x, _perspective(y, player))


def color_to_side(color, player):
    return player if color else -player


def side_to_color(side, player):
    return player == 1 if side == 1 else player == -1


def eq_boards(a, b):
    if a.has_legal_en_passant() != b.has_legal_en_passant():
        return False

    if a.has_legal_en_passant() and a.ep_square != b.ep_square:
        return False

    return a.board_fen() == b.board_fen() and \
        a.castling_rights == b.castling_rights and \
        a.turn == b.turn


def pcb_to_key(pcb):
    fen = pcb.fen()
    pcb, turn, castling, ep, _, _ = fen.split(' ')
    return ' '.join([pcb, turn, castling, ep])


def cur_player(pcb):
    return 1 if pcb.turn else -1


def _set_castling_rights(board, x, y, color):
    if color:
        print(board[y, x])
        assert board[y, x] == WHITE_ROOK
        board[y, x] = WHITE_ROOK_CASTLE
    else:
        assert board[y, x] == BLACK_ROOK
        board[y, x] = BLACK_ROOK_CASTLE


def _perspective(y, player):
    if player == 1:
        return 7 - y
    elif player == -1:
        return y
    else:
        raise Exception(player)
