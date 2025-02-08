import json
import math
from abc import abstractmethod
from typing import List, Dict, Optional, Union

import numpy as np
from haversine import Unit
from numpy.random import RandomState
import haversine as ha

from cabra.common.distance_helper import DistanceMode, Position
from cabra.common.math_util import normalize_scalar
from cabra.environment.data_structure import EnvResource, NodeResourceValue, NodesConfigType


class NotAllowedPermutationError(Exception):

    def __init__(self, resource: str, res_class: Optional[str] = None, error_type: Optional[str] = None, *args):
        if error_type is not None:
            if error_type == '>':
                message = f'Trying to add more resource then available in the node for resource: {resource}'
            else:
                message = f'Trying to remove more resource then available in the node for resource: {resource}'
        else:
            message = f'Trying to remove more resource then available in the node for resource: {resource}'
        if res_class is not None:
            message += f' and resource class: {res_class}'
        super(NotAllowedPermutationError, self).__init__(message, *args)


class DemandOutOfBoundError(Exception):

    def __init__(self, resource: str, value: Union[int, float], max_value: Union[int, float],
                 min_value: Union[int, float], *args):
        message = f'Adding a demand of {resource} out of resource bounds. ' \
                  f'Requested {value}, bounds are [{min_value}, {max_value}]'
        super(DemandOutOfBoundError, self).__init__(message, *args)


class NodeAbstract:

    def __init__(
            self,
            name: str,
            index: int,
            position: Position,
            node_features: List[str],
            zone: Optional[str] = None,
            distance_mode: DistanceMode = DistanceMode.L1
    ):
        self.name: str = name
        self.index: int = index
        self.position: Position = position
        self.original_position: Position = Position(lat=position.lat, lng=position.lng)
        self.node_features: List[str] = node_features
        self.zone: Optional[str] = zone
        self.distance_mode: DistanceMode = distance_mode

        for feature in self.node_features:
            assert feature in NODE_FEATURES_MAPPING, \
                f'node feature {feature} is not available. Available are: {list(NODE_FEATURES_MAPPING.keys())}'

    def __str__(self):
        if self.zone is not None:
            return f'<Node zone={self.zone} index={self.index} name={self.name} position={str(self.position)}>'
        else:
            return f'<Node index={self.index} name={self.name} position={str(self.position)}>'

    def distance(self, other: 'NodeAbstract') -> float:
        """
        Compute the distance between two nodes. It will compute the distance using the `distance_mode` property
        """
        return self.position.distance(other.position)

    def resource(self, name: str) -> Union[int, float]:
        """
        Returns the resource specified by `name`
        """
        return getattr(self, name)

    @abstractmethod
    def state_features(self, normalized: bool = False) -> List[float]:
        """
        Returns the list of features that are used for describing the single node.
        If `normalized` is True the values are normalized using the node config
        """
        pass

    @abstractmethod
    def allocation_sufficient(self, demand: float, resource: str) -> bool:
        """
        Verify if the node has `demand` units of its resource.
         If `demand` exceeds the total availability it raises DemandExceededError
        """
        pass

    @abstractmethod
    def set_capacity(self, next_capacity: float, resource: str, is_prediction: bool = False):
        """
        Update the node's features using the demand for the next episode.
        `next_demand` represents the number of units available in the next step.
        If `is_prediction` is True, the value will be associated to the prediction of demand
        If `next_demand` exceeds the total availability it raises DemandExceededError
        """
        pass

    @abstractmethod
    def allocate(self, quantity: int, resource: str):
        """
        Allocates `quantity` units of the resource. Method used for applying the action selected by the agent.
        If the quantity allocated exceed the maximum capacity it raises a NotAllowedPermutationError
        """
        pass

    @abstractmethod
    def allocation_possible(self, quantity: int, resource: str) -> bool:
        """
        Verifies if `quantity` units of the resource `resource` are available in the node
        """
        pass

    @abstractmethod
    def remove(self, quantity: int, resource: str):
        """
        Removes `quantity` units of the resource. Method used for applying the action selected by the agent.
        If the quantity is not available it raises a NotAllowedPermutationError
        """
        pass

    @abstractmethod
    def removal_possible(self, quantity: int, resource: str) -> bool:
        """
        Verifies if `quantity` units of the resource `resource` are removable from the node
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets the node to its initial level of capacity.
        """
        pass


class Node(NodeAbstract):

    def __init__(
            self,
            name: str,
            index: int,
            position: Position,
            n_bikes: int,
            total_slots: int,
            node_features: List[str],
            zone: Optional[str] = None,
            shortage_threshold: int = 0,
            critical_threshold: float = 0.2,
            critical_normalized: bool = True,
            predicted_demand: Optional[int] = None,
            use_prediction: bool = False,
            distance_mode: DistanceMode = DistanceMode.L1,
    ):
        super(Node, self).__init__(name, index, position, node_features, zone, distance_mode)
        self.bikes: int = n_bikes
        self.total_slots: int = total_slots
        self.ongoing_trips: int = 0
        self.shortage_threshold: int = shortage_threshold
        self.critical_threshold: float = critical_threshold
        self.critical_normalized: bool = critical_normalized
        self.bikes_prediction: Optional[int] = None
        self.empty_slots_prediction: Optional[int] = None
        self.ongoing_trips_prediction: Optional[int] = None
        self.use_prediction: bool = False
        self.use_prediction = use_prediction
        if predicted_demand is not None:
            self.use_prediction = True
            self.set_capacity(predicted_demand, is_prediction=True)
        self.initial_allocation: Dict[str, int] = {
            'bikes': n_bikes,
            'empty_slots': total_slots - n_bikes,
            'bikes_prediction': self.bikes_prediction,
            'empty_slots_prediction': self.empty_slots_prediction,
            'total_slots': total_slots
        }

    def __str__(self):
        string = super(Node, self).__str__()[:-1]
        string = f'{string} bikes={self.bikes} empty_slots={self.empty_slots} ongoing_trips={self.ongoing_trips}' \
                 f' total_slots={self.total_slots}>'
        if self.use_prediction:
            string = string[:-1]
            string = f'{string} bikes_pred={self.bikes_prediction} empty_slots_pred={self.empty_slots_prediction} ' \
                     f'ongoing_trips_prediction={self.ongoing_trips_prediction}>'
        return string

    @property
    def empty_slots(self):
        return self.total_slots - self.bikes

    @property
    def is_in_shortage(self) -> bool:
        if self.bikes <= self.shortage_threshold:
            return True
        if self.empty_slots <= self.shortage_threshold:
            return True
        return False

    @property
    def will_be_in_shortage(self) -> bool:
        if self.use_prediction:
            if self.bikes_prediction <= self.shortage_threshold:
                return True
            if self.empty_slots_prediction <= self.shortage_threshold:
                return True
        return False

    @property
    def state_feature_size(self) -> int:
        size = 0
        for feature in self.node_features:
            size += NODE_FEATURES_SIZE_MAPPING[feature]()

        if self.use_prediction:
            size *= 2
        return size

    def resource(self, name: str = 'bikes') -> Union[int, float]:
        return super(Node, self).resource(name)

    def state_features(self, normalized: bool = False) -> List[float]:
        features: List[float] = [
            NODE_FEATURES_MAPPING[feature](self, use_prediction=False, normalized=normalized)
            for feature in self.node_features if feature != 'position'
        ]
        if 'position' in self.node_features:
            features += NODE_FEATURES_MAPPING['position'](self, use_prediction=False, normalized=normalized)

        if self.use_prediction:
            for feature in self.node_features:
                features.append(
                    NODE_FEATURES_MAPPING[feature](self, use_prediction=True, normalized=normalized)
                )

        # shortage = 1 if self.is_in_shortage else 0
        # bikes = self.bikes
        # empty_slots = self.empty_slots
        # ongoing_trips = self.ongoing_trips
        # if normalized:
        #     bikes = normalize_scalar(bikes, max_val=self.total_slots, min_val=0)
        #     empty_slots = normalize_scalar(empty_slots, max_val=self.total_slots, min_val=0)
        #     ongoing_trips = normalize_scalar(ongoing_trips, max_val=self.total_slots, min_val=0)
        #
        # features: List[float] = [
        #     shortage,
        #     bikes,
        #     empty_slots,
        #     ongoing_trips,
        # ]
        # if self.use_prediction:
        #     will_shortage = 1 if self.will_be_in_shortage else 0
        #     bikes_pred = self.bikes_prediction
        #     empty_slots_pred = self.empty_slots_prediction
        #     ongoing_trips_pred = self.ongoing_trips_prediction
        #     if normalized:
        #         bikes_pred = normalize_scalar(bikes_pred, max_val=self.total_slots, min_val=0)
        #         empty_slots_pred = normalize_scalar(empty_slots_pred, max_val=self.total_slots, min_val=0)
        #         ongoing_trips_pred = normalize_scalar(ongoing_trips_pred, max_val=self.total_slots, min_val=0)
        #     features.append(will_shortage)
        #     features.append(bikes_pred)
        #     features.append(empty_slots_pred)
        #     features.append(ongoing_trips_pred)
        return features

    def allocation_sufficient(self, demand: float, resource: str = 'bikes') -> bool:
        return self.bikes >= demand

    def set_capacity(self, next_capacity: float, resource: str = 'bikes', is_prediction: bool = False):
        if next_capacity > self.total_slots or next_capacity < 0:
            raise DemandOutOfBoundError('bikes', next_capacity, self.total_slots, 0)
        if is_prediction is False:
            self.bikes = next_capacity
        else:
            self.bikes_prediction = next_capacity
            self.empty_slots_prediction = self.total_slots - self.bikes_prediction
            self.use_prediction = True

    def set_ongoing_trips(self, value: int, is_prediction: bool = False):
        if is_prediction is False:
            self.ongoing_trips = value
        else:
            self.ongoing_trips_prediction = value

    def allocate(self, quantity: int, resource: str = 'bikes'):
        if (self.bikes + quantity) > self.total_slots:
            raise NotAllowedPermutationError(resource, error_type='>')
        self.bikes += quantity

    def allocation_possible(self, quantity: int, resource: str = 'bikes') -> bool:
        return (self.bikes + quantity) <= self.total_slots

    def remove(self, quantity: int, resource: str = 'bikes'):
        if (self.bikes - quantity) < 0:
            raise NotAllowedPermutationError(resource, error_type='<')
        self.bikes -= quantity

    def removal_possible(self, quantity: int, resource: str = 'bikes') -> bool:
        return (self.bikes - quantity) >= 0

    def reset(self):
        self.bikes = self.initial_allocation['bikes']
        self.ongoing_trips = 0
        if self.use_prediction:
            self.bikes_prediction = self.initial_allocation['bikes_prediction']
            self.ongoing_trips_prediction = 0
            self.empty_slots_prediction = self.initial_allocation['empty_slots_prediction']

    def is_critical(self, critical_threshold: Optional[float] = None) -> bool:
        return self.is_empty_critical(critical_threshold) or self.is_full_critical(critical_threshold)

    def is_empty_critical(self, critical_threshold: Optional[float] = None) -> bool:
        threshold = critical_threshold if critical_threshold is not None else self.critical_threshold
        bikes = self.bikes
        if self.critical_normalized:
            bikes = normalize_scalar(bikes, min_val=0, max_val=self.total_slots)
        return bikes <= threshold

    def is_full_critical(self, critical_threshold: Optional[float] = None) -> bool:
        threshold = critical_threshold if critical_threshold is not None else self.critical_threshold
        empty_slots = self.empty_slots
        if self.critical_normalized:
            empty_slots = normalize_scalar(empty_slots, min_val=0, max_val=self.total_slots)
        return empty_slots <= threshold

    def fullness_ratio(self) -> float:
        return node_bikes(self, normalized=True)

    def copy(self) -> 'Node':
        n = Node(
            name=self.name,
            index=self.index,
            position=self.position,
            n_bikes=self.bikes,
            total_slots=self.total_slots,
            node_features=self.node_features,
            zone=self.zone,
            shortage_threshold=self.shortage_threshold,
            critical_threshold=self.critical_threshold,
            critical_normalized=self.critical_normalized,
            predicted_demand=None,
            use_prediction=False,
            distance_mode=self.distance_mode
        )
        n.initial_allocation = self.initial_allocation
        return n


def node_resources_to_dict(resources: List[EnvResource]) -> Dict[str, EnvResource]:
    return {r.name: r for r in resources}


def node_resources_to_node_resource_values_dict(resources: List[EnvResource]) -> Dict[str, NodeResourceValue]:
    return {r.name: NodeResourceValue(name=r.name,
                                      bikes=r.bikes_per_node) for r in resources}


def init_nodes_in_grid(
        n_nodes: int,
        total_slots: np.ndarray,
        bikes: np.ndarray,
        shortage_threshold: int,
        critical_threshold: float,
        critical_normalized: bool,
        node_features: List[str],
        distance_mode: DistanceMode,
        predicted_demand: Optional[List[int]] = None,
        use_prediction: bool = False,
) -> List['Node']:
    cols = math.ceil(math.sqrt(n_nodes))
    nodes: List['Node'] = []
    rows = math.ceil(n_nodes / cols)
    index = 0
    for row in range(rows):
        for col in range(cols):
            if index < n_nodes:
                nodes.append(
                    Node(f'n_{index}', index, position=Position(lat=row, lng=col),
                         n_bikes=bikes[index], total_slots=total_slots[index],
                         node_features=node_features,
                         shortage_threshold=shortage_threshold,
                         critical_threshold=critical_threshold,
                         critical_normalized=critical_normalized,
                         predicted_demand=predicted_demand[index] if predicted_demand is not None else None,
                         use_prediction=use_prediction,
                         distance_mode=distance_mode)
                )
            index += 1
    return nodes


def init_generated_nodes(
        nodes_config,
        random_state: RandomState,
        predicted_demand: Optional[List[int]] = None,
        use_prediction: bool = False,
) -> List['Node']:
    total_slots = np.floor(
        random_state.normal(
            nodes_config.generated_config.total_slots_avg,
            nodes_config.generated_config.total_slots_std,
            nodes_config.n_nodes
        )
    )
    bikes = np.floor(total_slots * nodes_config.generated_config.bikes_percentage)
    return init_nodes_in_grid(
        n_nodes=nodes_config.n_nodes,
        total_slots=total_slots,
        bikes=bikes,
        critical_threshold=nodes_config.critical_threshold,
        critical_normalized=nodes_config.critical_normalized,
        node_features=nodes_config.nodes_features,
        shortage_threshold=nodes_config.shortage_threshold,
        distance_mode=nodes_config.distance_mode,
        predicted_demand=predicted_demand,
        use_prediction=use_prediction
    )


def init_loaded_nodes(
        nodes_config,
        predicted_demand: Optional[List[int]] = None,
        use_prediction: bool = False
) -> List['Node']:
    nodes: List[Node] = []
    with open(nodes_config.loaded_config.get_nodes_load_path(), 'r') as f:
        nodes_data = json.load(f)
    bikes_percentage_from_data = nodes_config.loaded_config.bikes_percentage_from_data
    bikes_percentage = nodes_config.loaded_config.bikes_percentage
    node_index = 0
    for node_name, node_data in nodes_data['nodes'].items():
        percentage = node_data['bikes_percentage'] if bikes_percentage_from_data else bikes_percentage
        capacity = node_data['capacity']
        zone = node_data['zone_id'] if 'zone_id' in node_data else None
        bikes = round(capacity * percentage, 0)
        node_coordinates = node_data['position']['coordinates']
        nodes.append(Node(
            name=node_name,
            index=node_index,
            position=Position(lat=node_coordinates[1], lng=node_coordinates[0]),
            n_bikes=bikes, total_slots=capacity,
            shortage_threshold=nodes_config.shortage_threshold,
            critical_threshold=nodes_config.critical_threshold,
            critical_normalized=nodes_config.critical_normalized,
            node_features=nodes_config.nodes_features,
            zone=zone,
            distance_mode=nodes_config.distance_mode,
            predicted_demand=predicted_demand,
            use_prediction=use_prediction
        ))
        node_index += 1
    return nodes


def init_nodes(
        nodes_config,
        random_state: RandomState,
        predicted_demand: Optional[List[int]] = None,
        use_prediction: bool = False,
) -> List['Node']:
    if nodes_config.nodes_config == NodesConfigType.Generated:
        return init_generated_nodes(nodes_config, random_state, predicted_demand, use_prediction)
    if nodes_config.nodes_config == NodesConfigType.Loaded:
        return init_loaded_nodes(nodes_config, predicted_demand, use_prediction)


class DistancesProvider:

    def __init__(self, config):
        self.config = config
        self.nodes_config_path: Optional[str] = self.config.environment.nodes.loaded_config.get_nodes_load_path()
        self.zones_config_path: str = self.config.environment.nodes.loaded_config.get_zones_load_path()
        self.nodes_mode: NodesConfigType = self.config.environment.nodes.nodes_config
        self.distances: Dict[str, Dict[str, float]] = {}
        self.zones_distances: Dict[str, Dict[str, float]] = {}
        self.max_distance: Optional[float] = None
        self.zones_max_distance: Optional[float] = None
        self.max_position: Optional[Position] = None
        self.min_position: Optional[Position] = None
        self.load_distances()
        self.node_name_index_mapping: Dict[str, int] = {}

    def load_distances(self):
        if self.nodes_config_path is not None and self.nodes_mode == NodesConfigType.Loaded:
            with open(self.nodes_config_path, 'r') as f:
                data = json.load(f)

            for node, node_data in data['nodes'].items():
                node_coordinates = node_data['position']['coordinates']
                lat = node_coordinates[1]
                lng = node_coordinates[0]
                if self.max_position is None:
                    self.max_position = Position(lat=lat, lng=lng)
                else:
                    if lat > self.max_position.lat:
                        self.max_position.lat = lat
                    if lng > self.max_position.lng:
                        self.max_position.lng = lng
                if self.min_position is None:
                    self.min_position = Position(lat=lat, lng=lng)
                else:
                    if lat < self.min_position.lat:
                        self.min_position.lat = lat
                    if lng < self.min_position.lng:
                        self.min_position.lng = lng
                self.distances[node] = node_data['distances']

            with open(self.zones_config_path, 'r') as f:
                zones_data = json.load(f)

            for zone_id, zone_data in zones_data.items():
                self.zones_distances[zone_id] = zone_data['distances']

    def set_node_name_index_mapping(self, nodes: List[Node]):
        self.node_name_index_mapping = {n.name: i for i, n in enumerate(nodes)}

    def get_nodes_distance(self, start: Node, end: Node) -> float:
        if start.name in self.distances and end.name in self.distances[start.name]:
            return self.distances[start.name][end.name]
        else:
            distance = start.distance(end)
            if start.name in self.distances:
                self.distances[start.name][end.name] = distance
            else:
                self.distances[start.name] = {
                    end.name: distance
                }
            return distance

    def get_zones_distance(self, start_zone: str, end_zone: str) -> float:
        return self.zones_distances[start_zone][end_zone]

    def get_node_distances(self, node: Node, nodes: Optional[List[Node]] = None) -> List[float]:
        if node.name in self.distances and len(self.distances[node.name]) == (len(nodes) - 1):
            distances = self.distances[node.name]
            return list(distances.values())
        else:
            for other in nodes:
                _ = self.get_nodes_distance(node, other)
            distances = self.distances[node.name]
            return list(distances.values())

    def get_zone_distances(self, zone: str) -> List[float]:
        return list(self.zones_distances[zone].values())

    def get_max_distance(self, nodes: Optional[List[Node]] = None) -> float:
        if self.max_distance is not None:
            return self.max_distance
        max_distance = 0
        if self.nodes_config_path is not None and self.nodes_mode == NodesConfigType.Loaded:
            for node, node_distances in self.distances.items():
                for _, distance in self.distances[node].items():
                    if distance > max_distance:
                        max_distance = distance
        else:
            for n_i in nodes:
                for n_j in nodes:
                    distance = self.get_nodes_distance(n_i, n_j)
                    if distance > max_distance:
                        max_distance = distance
        self.max_distance = max_distance
        return max_distance

    def get_zones_max_distance(self) -> float:
        if self.zones_max_distance is not None:
            return self.zones_max_distance
        max_distance = 0
        for _, zone_distances in self.zones_distances.items():
            for _, distance in zone_distances.items():
                if distance > max_distance:
                    max_distance = distance
        self.zones_max_distance = max_distance
        return max_distance

    @staticmethod
    def point_distance_from_nodes(point: Position, nodes_positions: np.ndarray):
        point_vector = np.array([point.to_numpy()] * nodes_positions.shape[0])
        return ha.haversine_vector(point_vector, nodes_positions, unit=Unit.METERS)


def node_shortage_flag(node: 'Node', use_prediction: bool = False, **kwargs) -> float:
    if use_prediction:
        return 1 if node.will_be_in_shortage else 0
    else:
        return 1 if node.is_in_shortage else 0


def node_bikes(node: 'Node', use_prediction: bool = False, normalized: bool = True, **kwargs) -> float:
    if use_prediction:
        value = node.bikes_prediction
    else:
        value = node.bikes
    if normalized:
        return normalize_scalar(value, max_val=node.total_slots, min_val=0)
    else:
        return value


def node_empty_slots(node: 'Node', use_prediction: bool = False, normalized: bool = True, **kwargs) -> float:
    if use_prediction:
        value = node.empty_slots_prediction
    else:
        value = node.empty_slots
    if normalized:
        return normalize_scalar(value, max_val=node.total_slots, min_val=0)
    else:
        return value


def node_ongoing_trips(node: 'Node', use_prediction: bool = False, normalized: bool = True, **kwargs) -> float:
    if use_prediction:
        value = node.ongoing_trips_prediction
    else:
        value = node.ongoing_trips
    if normalized:
        return normalize_scalar(value, max_val=node.total_slots, min_val=0)
    else:
        return value


def node_bikes_critical_flag(node: 'Node', use_prediction: bool = False, **kwargs) -> float:
    if use_prediction:
        return 1 if node.bikes_prediction <= node.critical_threshold else 0
    else:
        return 1 if node.bikes <= node.critical_threshold else 0


def node_empty_critical_flag(node: 'Node', use_prediction: bool = False, **kwargs) -> float:
    if use_prediction:
        return 1 if node.empty_slots_prediction <= node.critical_threshold else 0
    else:
        return 1 if node.empty_slots <= node.critical_threshold else 0


def node_critical_flag(node: 'Node', use_prediction: bool = False, **kwargs) -> float:
    if node_bikes_critical_flag(node, use_prediction, **kwargs) == 1 \
            or node_empty_critical_flag(node, use_prediction, **kwargs) == 1:
        return 1
    else:
        return 0


def node_position(node: Node, use_prediction: bool = False, **kwargs) -> List[float]:
    return [node.position.lat, node.position.lng]


NODE_FEATURES_MAPPING = {
    'shortage_flag': node_shortage_flag,
    'bikes': node_bikes,
    'empty_slots': node_empty_slots,
    'ongoing_trips': node_ongoing_trips,
    'bikes_critic_flag': node_bikes_critical_flag,
    'empty_critic_flag': node_empty_critical_flag,
    'critical_flag': node_critical_flag,
    'position': node_position
}

NODE_FEATURES_SIZE_MAPPING = {
    'shortage_flag': lambda **kwargs: 1,
    'bikes': lambda **kwargs: 1,
    'empty_slots': lambda **kwargs: 1,
    'ongoing_trips': lambda **kwargs: 1,
    'bikes_critic_flag': lambda **kwargs: 1,
    'empty_critic_flag': lambda **kwargs: 1,
    'critical_flag': lambda **kwargs: 1,
    'position': lambda **kwargs: 2
}
