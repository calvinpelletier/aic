from aic.const import ELO_BINS


def elo_to_bin(elo: int):
    for i in range(len(ELO_BINS)):
        if elo < ELO_BINS[i]:
            return i
    return len(ELO_BINS)
