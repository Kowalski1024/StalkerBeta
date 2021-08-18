from .manager_base import ManagerBase
from sc2.unit import Unit
from sc2.units import Units
from sc2.position import Point2, Point3
from typing import Set, Dict, Tuple, Union
from matplotlib.patches import RegularPolygon
from sc2_math import Segment, LinearRing


class TownhallsManager(ManagerBase):
    def __init__(self):
        super().__init__()
        self.townhalls_tags: Dict[int, Unit] = dict()
        self.build_workers: Set[int] = set()

    async def update(self):
        pass

    async def post_update(self):
        pass


class Townhall:
    class MineralField:
        def __init__(self, resource: Unit, points: Tuple[Point2, Point2]):
            self.resource = resource
            self.workers_tag = set()
            self.gather_point, self.return_point = points
            self.mid_distance = self.gather_point.distance_to(self.return_point) / 1.2
            self.worker_amount = 0

        def add(self, unit: Unit):
            if self.worker_amount > 2:
                return False
            self.workers_tag.add(unit.tag)
            if unit.is_gathering:
                unit.gather(self.resource)
            self.worker_amount += 1
            return True

        def remove(self, unit: Union[Unit, int]):
            if isinstance(unit, Unit):
                unit = unit.tag
            self.workers_tag.remove(unit)
            self.worker_amount -= 1

    def __init__(self, mineral_fields: Units, bot, townhall: Unit):
        self.mineral_workers: Set[int] = set()
        self.townhall = townhall
        self.townhall_vertices = self.compute_townhall_vertices()
        self.mineral_fields = mineral_fields
        self.mineral_fields_dict: Dict[int, Townhall.MineralField] = {
            unit.tag: Townhall.MineralField(unit, self.compute_field_points(unit)) for unit in mineral_fields}
        self.bot = bot

    async def mining_debug(self):
        for field in self.mineral_fields_dict.values():
            p = Point3((field.gather_point.x, field.gather_point.y, self.townhall.position3d.z))
            self.bot.client.debug_sphere_out(p=p, r=0.1)
            p1 = Point3((field.return_point.x, field.return_point.y, self.townhall.position3d.z + 0.3))
            self.bot.client.debug_sphere_out(p=p1, r=0.1)
            self.bot.client.debug_line_out(p, p1)

    def compute_townhall_vertices(self) -> list:
        polygon = RegularPolygon(self.townhall.position_tuple, 40, 2.875)
        vertices = polygon.get_path().vertices.copy()
        vertices = polygon.get_patch_transform().transform(vertices)
        return list(map(Point2, vertices))

    def compute_field_points(self, field: Unit):
        def field_points(unit) -> list:
            x, y = unit.position_tuple
            return sorted([(x, y), (x + 0.5, y), (x - 0.5, y)], key=lambda pos: self.townhall.distance_to(pos))

        def vertices(unit) -> list:
            x, y = unit.position_tuple
            return [(x + 1, y + 0.5), (x + 1, y - 0.5), (x - 1, y + 0.5), (x - 1, y - 0.5)]

        def gather_vertices(unit, dis: float):
            x, y = unit.position_tuple
            x_dis = 1 + dis
            y_dis = 0.5 + dis
            return [(x + x_dis, y + 0.5), (x + x_dis, y - 0.5),
                    (x + 1, y - y_dis), (x - 1, y - y_dis),
                    (x - x_dis, y - 0.5), (x - x_dis, y + 0.5),
                    (x - 1, y + y_dis), (x + 1, y + y_dis)]

        field_points_list = field_points(field)
        neighbors = self.mineral_fields.closer_than(distance=2, position=field)
        neighbors.remove(field)  # remove self from the list
        for neighbor in neighbors:
            polygon = LinearRing(vertices(neighbor))
            for point in field_points(field):
                line = Segment(point, self.townhall.position_tuple)
                if polygon.intersects(line):
                    field_points_list.pop(0)
                else:
                    break
        if field.position_tuple not in field_points_list:
            field_point = field_points_list.pop(0)
        else:
            field_point = field.position_tuple
        line = Segment(field_point, self.townhall.position_tuple)
        rectangle = LinearRing(gather_vertices(field, dis=0.3125))
        intersection = rectangle.intersection(line).pop(0)
        gather_point = Point2((intersection[0], intersection[1]))
        return gather_point, gather_point.closest(self.townhall_vertices)

    async def speed_mining(self):
        for field in self.mineral_fields_dict.values():
            for worker_tag in list(field.workers_tag):
                worker: Unit = self.bot.units_dict[worker_tag]
                if worker.is_returning and worker.distance_to_squared(field.return_point) < field.mid_distance ** 2:
                    worker.move(field.return_point)
                    worker.smart(target=self.townhall, queue=True)
                    worker.smart(target=field.resource, queue=True)
                elif worker.is_gathering:
                    if worker.order_target != field.resource.tag and isinstance(worker.order_target, int):
                        new_field = self.mineral_fields_dict[worker.order_target]
                        if new_field.worker_amount < field.worker_amount and new_field.add(worker):
                            field.remove(worker)
                        else:
                            worker.gather(field.resource)
                    elif field.mid_distance ** 2 > worker.distance_to_squared(field.gather_point) > 1:
                        worker.move(field.gather_point)
                        worker.gather(target=field.resource, queue=True)
                        worker.return_resource(queue=True)

    def register_worker(self, worker: Unit) -> bool:
        for field in sorted(self.mineral_fields_dict.values(), key=lambda f: f.resource.distance_to(worker)):
            if field.add(worker):
                return True
        return False
