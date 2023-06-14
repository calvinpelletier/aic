from attrs import define

from aic.data.database.game.meta import Meta


@define
class CompressedGame:
    meta: Meta
    actions: list[int]
