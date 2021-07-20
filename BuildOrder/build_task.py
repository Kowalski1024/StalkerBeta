from sc2.constants import *


class BuildTask:
    def __init__(self, unit_id: UnitTypeId, priority, safe_resources=False):
        self.unit_id = unit_id
        self.priority = priority
        self.safe_resources = safe_resources
