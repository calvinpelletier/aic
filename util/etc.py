from aic.const import ELO_BINS, PLY_BINS


def elo_to_bin(elo: int):
    for i in range(len(ELO_BINS)):
        if elo < ELO_BINS[i]:
            return i
    return len(ELO_BINS)


def ply_to_bin(elo: int):
    for i in range(len(PLY_BINS)):
        if elo < PLY_BINS[i]:
            return i
    return len(PLY_BINS)
