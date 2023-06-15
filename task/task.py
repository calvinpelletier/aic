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


class BoardMetaHistory2OutcomeAction(Task):
    def _preprocess(s, data, game):
        pp.board(data, game)
        pp.meta(data, game)
        pp.history(data, game)
        pp.outcome(data, game)
        pp.action(data, game)


TASK_NAME_TO_CLS = {
    'b2l': Board2Legal,
    'bm2o': BoardMeta2Outcome,
    'bmh2oa': BoardMetaHistory2OutcomeAction,
}

def build_task(cfg):
    return TASK_NAME_TO_CLS[cfg.task]()
