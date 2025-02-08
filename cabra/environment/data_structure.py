from dataclasses import dataclass
from enum import Enum

from cabra.common.enum_utils import ExtendedEnum


@dataclass
class EnvResource:
    name: str
    total_slots: int
    total_bikes: int
    slots_per_node: int
    bikes_per_node: int


class ActionType(str, ExtendedEnum):
    Wait = 'wait'
    Target = 'target'
    Quantity = 'quantity'
    Zone = 'zone'


@dataclass
class NodeResourceValue:
    name: str
    bikes: int


class StateFeatureName(str, ExtendedEnum):
    NodeFeatures = 'node-features'
    TruckFeatures = 'truck-features'
    CurrentTime = 'current-time'
    DatasetTime = 'dataset-time'
    TargetNode = 'target-node'
    CurrentTruck = 'current-truck'
    CurrentTruckFull = 'current-truck-full'
    PreviousState = 'previous-state'
    PreviousAction = 'previous-action'
    Zones = 'zones'
    SelectedZone = 'selected-zone'
    CurrentZone = 'current-zone'
    Weather = 'weather'
    PreviousZoneAction = 'previous-zone-action'
    PreviousFullAction = 'previous-full-action'


class RewardFunctionType(str, ExtendedEnum):
    GlobalShortageAndCost = 'global-shortage-and-cost'


class ReplacementPolicy(str, ExtendedEnum):
    Random = 'Random'
    FIFO = 'Fifo'


class EpsilonType(str, ExtendedEnum):
    Scalar = 'scalar'
    LinearDecay = 'linear-decay'
    ExponentialDecay = 'exponential-decay'


class NodeTypeDistribution(str, ExtendedEnum):
    Random = 'random'
    Equally = 'equally'


class NodesConfigType(str, ExtendedEnum):
    Generated = 'generated'
    Loaded = 'loaded'


class NetworkType(str, ExtendedEnum):
    FullyConnected = 'fully-connected'
    FullyConnectedVariable = 'fully-connected-variable'
    FullyConnectedDiamond = 'fully-connected-diamond'
    Dueling = 'dueling'


class StateType(str, ExtendedEnum):
    Target = 'target_state'
    Quantity = 'quantity_state'
    Zone = 'zone'


class DiscretizeMode(str, ExtendedEnum):
    Greedy = 'greedy'
    Proportional = 'proportional'


class SubAgentType(str, ExtendedEnum):
    Target = 'target_sub_agent'
    Quantity = 'quantity_sub_agent'

    def get_action_type(self) -> ActionType:
        mapping = {
            SubAgentType.Target: ActionType.Target,
            SubAgentType.Quantity: ActionType.Quantity,
        }
        return mapping[self]

    def get_state_type(self) -> StateType:
        mapping = {
            SubAgentType.Target: StateType.Target,
            SubAgentType.Quantity: StateType.Quantity,
        }
        return mapping[self]


class DistributionType(str, Enum):
    Categorical = 'categorical'
    MultiCategorical = 'multi-categorical'
    Gaussian = 'gaussian'
    MultiBinary = 'multi-binary'
    GaussianCategorical = 'gaussian-categorical'
