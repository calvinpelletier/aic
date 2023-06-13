

PGN_GAME_START = '[Event "'


def pgn_iterator(reader):
    lines = []
    for line in reader:
        if line.startswith(PGN_GAME_START) and lines:
            yield ''.join(lines)
            lines = []
        lines.append(line)
    yield ''.join(lines)
