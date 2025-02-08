import math
from dataclasses import dataclass

import numpy as np
from haversine import haversine, Unit

from cabra.common.enum_utils import ExtendedEnum


class DistanceMode(str, ExtendedEnum):
    L1 = 'l1'
    L2 = 'l2'
    GeoDistance = 'geo-distance'
    StreetDistance = 'street-distance'


def l1_distance(a: 'Position', b: 'Position') -> float:
    return abs(a.lat - b.lat) + abs(a.lng - b.lng)


def l2_distance(a: 'Position', b: 'Position') -> float:
    return math.sqrt((a.lat - b.lat) ** 2 + (a.lng - b.lng) ** 2)


def geo_distance(a: 'Position', b: 'Position') -> float:
    return haversine(
        (a.lat, a.lng),
        (b.lat, b.lng),
        Unit.METERS
    )


MAPPING = {
    DistanceMode.L1: l1_distance,
    DistanceMode.L2: l2_distance,
    DistanceMode.GeoDistance: geo_distance
}


def get_distance(mode: DistanceMode, a: 'Position', b: 'Position') -> float:
    if mode in MAPPING:
        distance_fn = MAPPING[mode]
        return distance_fn(a, b)
    else:
        raise AttributeError(f'Distance Mode {mode} not available')


@dataclass
class Position:
    lat: float
    lng: float

    def __str__(self):
        return f'<lat={self.lat} lng={self.lng}>'

    def __eq__(self, other):
        return self.distance(other) == 0

    def distance(self, other: 'Position', mode: DistanceMode = DistanceMode.L1) -> float:
        return get_distance(mode, self, other)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.lng, self.lat], dtype=np.float)
