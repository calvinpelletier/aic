from aic.data.dataset import preprocess as pp


class Task:
    def preprocess(s, game):
        data = {}
        s._preprocess(data, game)
        return data

    def _preprocess(s, data, game):
        raise NotImplementedError()


class Board2Action(Task):
    def _preprocess(s, data, game):
        data['board'] = pp.board(game)
        data['action'] = pp.action(game)


class Board2Legal(Task):
    def _preprocess(s, data, game):
        data['board'] = pp.board(game)
        data['legal'] = pp.legal(game)


class BoardMeta2Outcome(Task):
    def _preprocess(s, data, game):
        data['board'] = pp.board(game)
        data['meta'] = pp.meta(game)
        data['outcome'] = pp.outcome(game)


class BoardMetaHistory2OutcomeAction(Task):
    def _preprocess(s, data, game):
        data['board'] = pp.board(game)
        data['meta'] = pp.meta(game)
        data['history'], data['history_len'] = pp.history(game)
        data['outcome'] = pp.outcome(game)
        data['action'] = pp.action(game)


TASK_NAME_TO_CLS = {
    'b2a': Board2Action,
    'b2l': Board2Legal,
    'bm2o': BoardMeta2Outcome,
    'bmh2oa': BoardMetaHistory2OutcomeAction,
}

def build_task(name):
    return TASK_NAME_TO_CLS[name]()
