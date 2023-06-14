from numpy.typing import ArrayLike


class Meta:
    @classmethod
    def from_list(cls, data: ArrayLike):
        return cls(*data)

    def as_list(s) -> list[int]:
        raise NotImplementedError()

    def __eq__(a, b):
        return type(a) == type(b) and a.as_list() == b.as_list()


class LichessMeta(Meta):
    def __init__(s,
        outcome: int,
        white_elo: int,
        black_elo: int,
        tc_base: int,
        tc_inc: int,
        termination: int,
        truncated: int,
    ):
        s.outcome = outcome
        s.white_elo = white_elo
        s.black_elo = black_elo
        s.tc_base = tc_base
        s.tc_inc = tc_inc
        s.termination = termination
        s.truncated = truncated

    def as_list(s) -> list[int]:
        return [
            s.outcome,
            s.white_elo,
            s.black_elo,
            s.tc_base,
            s.tc_inc,
            s.termination,
            s.truncated,
        ]
