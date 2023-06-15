

ELO_BINS = [1500, 1750, 2000, 2250, 2500]


def elo_to_bin(elo: int):
    for i in range(len(ELO_BINS)):
        if elo < ELO_BINS[i]:
            return i
    return len(ELO_BINS)
