from typing import Optional, List, Tuple

from numpy.random import RandomState

from cabra import SingleRunConfig
from cabra.common.math_util import normalize_scalar
from cabra.core.step import Step
from cabra.environment.node import Node, DistancesProvider


class NotAllowedDropReposition(Exception):

    def __init__(self, quantity: int, empty_slots: int, *args):
        message = f'Impossible to drop {quantity} bikes on node with {empty_slots} empty slots'
        super(NotAllowedDropReposition, self).__init__(message, *args)


class NotAllowedPickReposition(Exception):

    def __init__(self, quantity: int, empty_slots: int, *args):
        message = f'Impossible to pick {quantity} bikes for truck with {empty_slots} empty slots'
        super(NotAllowedPickReposition, self).__init__(message, *args)


class Truck:

    def __init__(
            self,
            name: str,
            index: int,
            capacity: int,
            initial_node: Node,
            truck_features: List[str],
            distances_provider: DistancesProvider,
            random_state: RandomState,
            total_trucks: int = 2,
            initial_load: int = 0,
            move_speed_avg: float = 18 / 3.6,  # in meter/second, so we divide 40 k/h by 3.6
            move_speed_std: float = 2,
            reposition_time_avg: float = 60,  # seconds for a single reposition
            reposition_time_std: float = 0.5,
            step_size: int = 600,  # step size of the simulation, used to scale to the timestep the reposition time
    ):
        # truck properties
        self.name: str = name
        self.index: int = index
        self.total_trucks: int = total_trucks
        self.truck_features: List[str] = truck_features
        self.initial_load: int = initial_load
        self.initial_node: Node = initial_node
        self.capacity: int = capacity
        self.move_speed_avg: float = move_speed_avg
        self.move_speed_std: float = move_speed_std
        self.reposition_time_avg: float = reposition_time_avg
        self.reposition_reposition_std: float = reposition_time_std
        self.step_size: int = step_size
        for feature in self.truck_features:
            assert feature in TRUCK_FEATURES_MAPPING, \
                f'truck feature {feature} is not available. Available are: {list(TRUCK_FEATURES_MAPPING.keys())}'
        # other properties
        self.distance_provider: DistancesProvider = distances_provider
        self.random: RandomState = random_state
        self.max_distance_time: int = int(round(self.distance_provider.max_distance / self.move_speed_avg, 0))
        self.max_relocation_time: int = int(round(self.capacity * self.reposition_time_avg, 0))
        self.max_reposition_time: int = self.max_distance_time + self.max_relocation_time
        # mutable properties
        self.current_node: Node = initial_node
        self.load: int = initial_load
        self.is_busy: bool = False
        self.reposition_start_step: Optional[Step] = None
        self.reposition_distance_time: Optional[int] = None
        self.reposition_relocate_time: Optional[int] = None
        self.reposition_time: Optional[int] = None

    def __str__(self):
        string = f'<Truck index={self.index} name={self.name} load={self.load} ' \
                 f'empty_slots={self.empty_slots} capacity={self.capacity} current_node={str(self.current_node)}>'
        return string

    @property
    def empty_slots(self) -> int:
        return self.capacity - self.load

    def is_idle(self, step: Step) -> bool:
        if self.reposition_start_step is None:
            return True
        step_total = step.total_steps
        busy_time = self.reposition_start_step.total_steps + self.reposition_time
        return step_total > busy_time

    def reset(self):
        self.load = self.initial_load
        self.current_node = self.initial_node
        self.is_busy = False
        self.reposition_start_step: Optional[Step] = None
        self.reposition_distance_time: Optional[int] = None
        self.reposition_relocate_time: Optional[int] = None
        self.reposition_time: Optional[int] = None

    def reposition(self, target_node: Node, quantity: int, step: Step):
        self.reposition_start_step = step
        # get the estimated time for the reposition
        self.reposition_time, self.reposition_distance_time, self.reposition_relocate_time = \
            self.estimate_total_reposition_time(target_node, quantity)
        # move the truck
        self.move(target_node)
        if quantity < 0:
            # quantity lower than 0, drop quantity bikes
            self.drop(quantity)
        else:
            # quantity greater than 0, pick quantity bikes
            self.pick(quantity)

    def move(self, target_node: Node):
        self.current_node = target_node

    def drop(self, quantity: int):
        if self.drop_possible(self.current_node, quantity):
            self.current_node.allocate(abs(quantity))
            self.load -= abs(quantity)
        else:
            raise NotAllowedDropReposition(quantity, self.current_node.empty_slots)

    def drop_possible(self, node: Node, quantity: int):
        return self.load >= abs(quantity) and node.allocation_possible(abs(quantity))

    def pick(self, quantity: int):
        if self.pick_possible(self.current_node, quantity):
            self.load += quantity
            self.current_node.remove(quantity)
        else:
            raise NotAllowedPickReposition(quantity, self.empty_slots)

    def pick_possible(self, node: Node, quantity: int):
        return (self.load + quantity) <= self.capacity and node.removal_possible(quantity)

    def sample_move_speed(self):
        # truncated normal distribution, so the value is at least 0.1 meter/second
        return max(0.1, self.random.normal(self.move_speed_avg, self.move_speed_std))

    def sample_reposition_time(self):
        # truncated normal distribution, so the value is always greater than 0
        return max(0, self.random.normal(self.reposition_time_avg, self.reposition_reposition_std))

    def estimate_total_reposition_time(self, target_node: Node, quantity: int) -> Tuple[int, int, int]:
        distance = self.distance_provider.get_nodes_distance(start=self.current_node, end=target_node)
        move_speed = self.sample_move_speed()
        # move speed is in meter/second
        distance_time = int(round(distance / move_speed, 0))

        # reposition speed is one bike is repositioned in reposition_time seconds
        reposition_time = self.sample_reposition_time()
        total_reposition_time = int(round(abs(reposition_time * quantity), 0))

        total_estimate_steps = distance_time + total_reposition_time
        return total_estimate_steps, distance_time, total_reposition_time

    def state_features(self, step: Step, normalized: bool = False) -> List[float]:
        features: List[float] = []
        for feature in self.truck_features:
            value = TRUCK_FEATURES_MAPPING[feature](self, step=step, normalized=normalized)
            if isinstance(value, list):
                features += value
            else:
                features.append(value)
        return features

    @property
    def state_feature_size(self) -> int:
        size = 0
        for feature in self.truck_features:
            size += TRUCK_FEATURES_SIZE_MAPPING[feature](self)

        return size

    def copy(self) -> 'Truck':
        t = Truck(
            name=self.name,
            index=self.index,
            capacity=self.capacity,
            initial_node=self.initial_node,
            truck_features=self.truck_features,
            distances_provider=self.distance_provider,
            random_state=self.random,
            total_trucks=self.total_trucks,
            initial_load=self.initial_load,
            move_speed_avg=self.move_speed_avg,
            move_speed_std=self.move_speed_std,
            reposition_time_avg=self.reposition_time_avg,
            reposition_time_std=self.reposition_reposition_std,
            step_size=self.step_size
        )
        t.current_node = self.current_node
        t.load = self.load
        t.is_busy = self.is_busy
        t.reposition_start_step = self.reposition_start_step
        t.reposition_distance_time = self.reposition_distance_time
        t.reposition_relocate_time = self.reposition_relocate_time
        t.reposition_time = self.reposition_time
        return t


def truck_position_feature(truck: Truck, normalized: bool = True, **kwargs) -> List[float]:
    # get position from current node position coordinates
    return [
        truck.current_node.position.lat, truck.current_node.position.lng
    ]


def truck_load_feature(truck: Truck, normalized: bool = True, **kwargs) -> float:
    if normalized:
        return normalize_scalar(value=truck.load, max_val=truck.capacity, min_val=0)
    else:
        return truck.load


def truck_busy_feature(truck: Truck, step: Step, normalized: bool = True, **kwargs) -> float:
    return 0 if truck.is_idle(step) else 1


TRUCK_FEATURES_MAPPING = {
    'position': truck_position_feature,
    'load': truck_load_feature,
    'busy': truck_busy_feature,
}


def truck_position_feature_size(truck, **kwargs) -> int:
    return 2
    # return len(truck.distance_provider.node_name_index_mapping)


TRUCK_FEATURES_SIZE_MAPPING = {
    'position': truck_position_feature_size,
    'load': lambda *args, **kwargs: 1,
    'busy': lambda *args, **kwargs: 1,
}


def init_trucks(
        config: SingleRunConfig,
        random_state: RandomState,
        initial_node: Node,
        distances_provider: DistancesProvider,
) -> List[Truck]:
    truck_config = config.environment.trucks
    trucks = [
        Truck(
            name=f'truck_{i}',
            index=i,
            capacity=truck_config.capacity,
            initial_node=initial_node,
            truck_features=truck_config.truck_features,
            distances_provider=distances_provider,
            random_state=random_state,
            total_trucks=truck_config.n_trucks,
            initial_load=truck_config.initial_load_level,
            move_speed_avg=truck_config.move_speed_avg,
            move_speed_std=truck_config.move_speed_std,
            reposition_time_avg=truck_config.reposition_time_avg,
            reposition_time_std=truck_config.reposition_time_std,
            step_size=config.environment.time_step.step_size,
        )
        for i in range(truck_config.n_trucks)
    ]
    return trucks


class TrucksWrapper:

    def __init__(
            self,
            trucks: List[Truck],
            config: SingleRunConfig
    ):
        self.config: SingleRunConfig = config
        self.trucks: List[Truck] = trucks

    def idle_trucks(self, step: Step) -> List[Truck]:
        idle_trucks = [truck for truck in self.trucks if truck.is_idle(step)]
        return idle_trucks

    def are_trucks_idle(self, step: Step) -> bool:
        return len(self.idle_trucks(step)) > 0

    def __len__(self):
        return len(self.trucks)
