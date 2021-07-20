from sc2.bot_ai import BotAI
from sc2.constants import *
from .build_task import BuildTask

from typing import Dict


class Builder:
    MAX_WORKERS = 80

    def __init__(self, bot: BotAI, order_dict: Dict[UnitTypeId, BuildTask]):
        self._bot = bot
        self._order_dict = order_dict

