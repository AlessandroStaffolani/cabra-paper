import itertools
from typing import Optional, List

from cabra.common.distance_helper import Position, DistanceMode
from cabra.environment.node import Node


class Zone:

    def __init__(self, name: str, nodes: Optional[List[Node]] = None, centroid: Optional[Position] = None):
        self.name: str = name
        self.nodes: List[Node] = nodes if nodes is not None else []
        self.centroid: Position = centroid
        self.max_zone_size: int = len(self.nodes)

    def __str__(self):
        return f'<Zone name={self.name} size={len(self)}>'

    def __len__(self):
        return len(self.nodes)

    def add_node(self, node: Node):
        self.nodes.append(node)
        self.max_zone_size += 1

    def distance(self, position: Position) -> Optional[float]:
        if self.centroid is not None:
            return self.centroid.distance(position, mode=DistanceMode.GeoDistance)
        else:
            return None

    def state_features(self, normalized: bool = True, **kwargs) -> List[float]:
        state_size = self.state_features_size
        state_features = list(itertools.chain.from_iterable([node.state_features(normalized) for node in self.nodes]))
        if len(state_features) < state_size:
            state_features += [0] * (state_size - len(state_features))
        return state_features

    @property
    def state_features_size(self) -> int:
        return self.nodes[0].state_feature_size * self.max_zone_size
