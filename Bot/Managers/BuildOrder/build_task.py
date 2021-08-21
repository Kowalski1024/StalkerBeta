from sc2.constants import *
from sc2.unit import Unit
from sc2.position import Point2


class BuildTask:
    def __init__(self, unit_id: UnitTypeId, priority, claim_resources=True):
        self.unit_id = unit_id
        self.priority = priority
        self.claim_resources = claim_resources
        self.target_position = None

