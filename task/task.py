from aic.data.dataset import preprocess as pp


class Task:
    def preprocess(s, game):
        data = {}
        s._preprocess(data, game)
        return data

    def _preprocess(s, data, game):
        raise NotImplementedError()


class Board2Legal(Task):
    def _preprocess(s, data, game):
        pp.board(data, game)
        pp.legal(data, game)


class BoardMeta2Outcome(Task):
    def _preprocess(s, data, game):
        pp.board(data, game)
        pp.meta(data, game)
        pp.outcome(data, game)


TASK_MAP = {
    'b2l': Board2Legal,
    'bm2o': BoardMeta2Outcome,
}

def build_task(cfg):
    return TASK_MAP[cfg.task]()
