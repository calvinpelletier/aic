from attrs import define

from aic.data.db.game.meta import Meta


@define
class CompressedGame:
    meta: Meta
    actions: list[int]
