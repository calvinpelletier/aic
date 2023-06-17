import chess

from aic.util import action_to_move


class Game:
    def __init__(s, meta, actions, rand_trim_start=None, rand_trim_end=None):
        s.outcome = meta.outcome
        s.white_elo = meta.white_elo
        s.black_elo = meta.black_elo
        s.tc_base = meta.tc_base
        s.tc_inc = meta.tc_inc

        s.actions = actions

        s.pcb = chess.Board()

        s._end = len(s.actions)
        if rand_trim_end is not None:
            x, y = rand_trim_end
            y = min(y, len(s.actions))
            if y > x:
                s._end = random.randrange(x, y)

        s.i = 0
        if rand_trim_start is not None:
            x, y = rand_trim_start
            y = min(y, s._end - 1)
            if y > x:
                trim = random.randrange(x, y)
                for _ in range(trim):
                    done = s.next_position()
                    assert not done
                assert s.i == trim

    def next_position(s):
        s.pcb.push(action_to_move(s.actions[s.i], s.pcb))
        s.i += 1
        return s.i == s._end
