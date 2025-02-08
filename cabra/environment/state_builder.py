import json
from typing import List, Optional, Dict, Any, Union

import numpy as np

from cabra.core.state import State, StateFeatureValue
from cabra.core.step import Step, STEP_UNITS_NORMALIZATION_FACTOR
from cabra.core.step_data import StepData
from cabra.common.math_util import normalize_scalar
from cabra.environment.action_space import RepositionAction, RepositionActionSpace, ZoneAction, \
    ZoneActionSpace
from cabra.environment.data_structure import StateFeatureName, StateType
from cabra.environment.node import Node, DistancesProvider
from cabra.environment.truck import Truck
from cabra.environment.zone import Zone

STACKED_STATE_FEATURES = []


def normalize_values(values: List[float],
                     max_val: float, min_val: float = 0, a: float = 0, b: float = 1) -> List[float]:
    np_values = np.array(values, dtype=np.float)
    np_result = (b - a) * ((np_values - min_val) / (max_val - min_val)) + a
    return np_result.tolist()


def _build_feature_node_features(
        nodes: List[Node],
        normalized: bool,
        **kwargs
) -> List[StateFeatureValue]:
    values = []
    for node in nodes:
        values += node.state_features(normalized=normalized)
    return [StateFeatureValue(name=StateFeatureName.NodeFeatures.value, value=values)]


def _build_feature_truck_features(
        trucks: List[Truck],
        current_step: Step,
        normalized: bool,
        **kwargs
) -> List[StateFeatureValue]:
    values = []
    for truck in trucks:
        values += truck.state_features(step=current_step, normalized=normalized)
    return [StateFeatureValue(name=StateFeatureName.TruckFeatures.value, value=values)]


def _build_one_hot_encoding_feature(size: int, feature_name, index: Optional[int] = None) -> StateFeatureValue:
    values = [0] * size
    if index is not None:
        values[index] = 1
    return StateFeatureValue(name=feature_name, value=values)


def _build_feature_current_time(current_step: Step,
                                additional_properties: Dict[str, Any],
                                normalized: bool,
                                **kwargs) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    if 'units_to_skip' not in additional_properties:
        raise AttributeError(f'State Feature {StateFeatureName.CurrentTime} requested '
                             f'but units_to_skip not present in additional_properties')
    units_to_skip = additional_properties['units_to_skip']
    values = []
    for unit, value in current_step.to_dict().items():
        if unit not in units_to_skip:
            unit_values = np.zeros(STEP_UNITS_NORMALIZATION_FACTOR[unit] + 1)
            unit_values[value] = 1
            values += unit_values.tolist()
    features.append(
        StateFeatureValue(name=f'{StateFeatureName.CurrentTime.value}', value=values)
    )
    return features


def _build_feature_dataset_time(
        current_demand: StepData,
        additional_properties: Dict[str, Any],
        **kwargs
) -> List[StateFeatureValue]:
    features: List[StateFeatureValue] = []
    if 'units_to_skip' not in additional_properties:
        raise AttributeError(f'State Feature {StateFeatureName.DatasetTime} requested '
                             f'but units_to_skip not present in additional_properties')
    units_to_skip = additional_properties['units_to_skip']
    assert 'second_step' in units_to_skip and 'week' in units_to_skip, \
        f'{StateFeatureName.DatasetTime} feature does not support second_step and week unit'
    values = []
    dataset_date = current_demand.date

    for unit_name, unit_max_value in STEP_UNITS_NORMALIZATION_FACTOR.items():
        unit_max_value = unit_max_value + 1
        if unit_name not in units_to_skip:
            unit_values = np.zeros(unit_max_value)
            if unit_name == 'week_day':
                index = dataset_date.weekday()
            elif unit_name == 'month':
                index = getattr(dataset_date, unit_name) - 1
            else:
                index = getattr(dataset_date, unit_name)
            unit_values[index] = 1
            values += unit_values.tolist()

    features.append(
        StateFeatureValue(name=f'{StateFeatureName.DatasetTime.value}', value=values)
    )
    return features


def _build_feature_node_target(
        nodes: List[Node], **kwargs) -> List[StateFeatureValue]:
    feature: StateFeatureValue = _build_one_hot_encoding_feature(
        size=len(nodes),
        feature_name=StateFeatureName.TargetNode.value,
    )
    return [feature]


def _build_feature_current_truck(
        trucks: List[Truck], **kwargs) -> List[StateFeatureValue]:
    feature: StateFeatureValue = _build_one_hot_encoding_feature(
        size=len(trucks),
        feature_name=StateFeatureName.CurrentTruck.value,
    )
    return [feature]


def _build_feature_current_truck_full(
        trucks: List[Truck],
        truck_index: int,
        current_step: Step,
        normalized: bool,
        **kwargs,
) -> List[StateFeatureValue]:
    return [
        StateFeatureValue(
            name=StateFeatureName.CurrentTruckFull.value,
            value=trucks[truck_index].state_features(current_step, normalized)
        )
    ]


def _build_feature_previous_state(
        state_size: int = 0,
        previous_state: Optional[State] = None,
        **kwargs
) -> List[StateFeatureValue]:
    if previous_state is None:
        values = np.zeros(state_size)
    else:
        values = previous_state
    return [
        StateFeatureValue(name=str(StateFeatureName.PreviousState.value), value=values)
    ]


def _build_feature_previous_action(
        normalized: bool,
        action_space: RepositionActionSpace,
        previous_action: Optional[RepositionAction] = None,
        **kwargs
) -> List[StateFeatureValue]:
    if previous_action is None:
        values = [0, 0]
    else:
        values = previous_action.to_tuple()
    if normalized:
        values[0] = normalize_scalar(values[0], max_val=action_space.target_space.size(), min_val=0)
        values[1] = normalize_scalar(values[1], max_val=action_space.quantity_space.size(), min_val=0)
    return [
        StateFeatureValue(name=str(StateFeatureName.PreviousAction.value), value=values)
    ]


def _build_feature_previous_action_vectorized(
        action_space: RepositionActionSpace,
        previous_action: Optional[RepositionAction] = None,
        **kwargs
) -> List[StateFeatureValue]:
    target_values = [0] * action_space.target_space.size()
    quantity_values = [0] * action_space.quantity_space.size()
    if previous_action is not None:
        target_values[previous_action.target] = 1
        quantity_values[previous_action.quantity] = 1
    return [
        StateFeatureValue(name=str(StateFeatureName.PreviousAction.value), value=target_values + quantity_values)
    ]


def _build_feature_weather(
        current_demand: StepData,
        normalized: bool,
        weather_info: Dict[str, Any],
        **kwargs
) -> List[StateFeatureValue]:
    assert current_demand.weather is not None, 'Weather feature requested, but weather not provided in dataset data'
    weather_data = current_demand.weather
    values = [0] * len(weather_info['conditions'])
    if weather_data.condition is not None:
        values[weather_info['conditions'].index(weather_data.condition)] = 1
    if weather_data.temperature is not None:
        temp = weather_data.temperature
    else:
        temp = weather_info['stats']['minTemperature']
    if weather_data.wind_speed is not None:
        wind_speed = weather_data.wind_speed
    else:
        wind_speed = weather_info['stats']['minWindSpeed']
    if normalized:
        temp = normalize_scalar(temp,
                                min_val=weather_info['stats']['minTemperature'],
                                max_val=weather_info['stats']['maxTemperature'])
        wind_speed = normalize_scalar(wind_speed,
                                      min_val=weather_info['stats']['minWindSpeed'],
                                      max_val=weather_info['stats']['maxWindSpeed'])
    values += [temp, wind_speed]
    return [
        StateFeatureValue(name=str(StateFeatureName.Weather.value), value=values)
    ]


def _build_feature_zones(
        zones: Dict[str, Zone],
        normalized: bool,
        **kwargs,
) -> List[StateFeatureValue]:
    values = []
    for _, zone in zones.items():
        values += zone.state_features(normalized=normalized)
    return [StateFeatureValue(name=StateFeatureName.Zones.value, value=values)]


def _build_feature_current_zone(
        zones: Dict[str, Zone],
        current_zone_id: Optional[str],
        normalized: bool,
        **kwargs,
) -> List[StateFeatureValue]:
    if current_zone_id is None:
        zone_size = zones[list(zones.keys())[0]].state_features_size
        values = [0] * zone_size
    else:
        values = zones[current_zone_id].state_features(normalized)
    return [StateFeatureValue(name=StateFeatureName.CurrentZone.value, value=values)]


def _build_feature_previous_zone_action(
        zone_action_space: ZoneActionSpace,
        previous_zone_action: Optional[ZoneAction] = None,
        **kwargs
) -> List[StateFeatureValue]:
    zone_values = [0] * zone_action_space.zone_space.size()
    if previous_zone_action is not None:
        zone_values[previous_zone_action.zone] = 1
    return [
        StateFeatureValue(
            name=str(StateFeatureName.PreviousZoneAction.value),
            value=zone_values)
    ]


def _build_feature_previous_full_action(
        zone_action_space: ZoneActionSpace,
        action_space: RepositionActionSpace,
        previous_zone_action: Optional[ZoneAction] = None,
        previous_action: Optional[RepositionAction] = None,
        **kwargs,
) -> List[StateFeatureValue]:
    previous_zone_feature = _build_feature_previous_zone_action(zone_action_space, previous_zone_action, **kwargs)
    previous_action_feature = _build_feature_previous_action_vectorized(action_space, previous_action, **kwargs)
    return [
        StateFeatureValue(
            name=str(StateFeatureName.PreviousFullAction.value),
            value=previous_zone_feature[0].value + previous_action_feature[0].value)
    ]


def _build_feature_selected_zone(
        zones: Dict[str, Zone],
        current_zone_id: Optional[str],
        **kwargs,
) -> List[StateFeatureValue]:
    index = None
    if current_zone_id is not None:
        tmp_index = 0
        for zone_id in zones.keys():
            if zone_id == current_zone_id:
                index = tmp_index
            tmp_index += 1

    return [
        _build_one_hot_encoding_feature(
            size=len(zones), feature_name=StateFeatureName.SelectedZone.value, index=index
        )
    ]


FEATURE_BUILD_MAPPING = {
    StateFeatureName.NodeFeatures: _build_feature_node_features,
    StateFeatureName.TruckFeatures: _build_feature_truck_features,
    StateFeatureName.CurrentTime: _build_feature_current_time,
    StateFeatureName.DatasetTime: _build_feature_dataset_time,
    StateFeatureName.TargetNode: _build_feature_node_target,
    StateFeatureName.CurrentTruck: _build_feature_current_truck,
    StateFeatureName.CurrentTruckFull: _build_feature_current_truck_full,
    StateFeatureName.PreviousState: _build_feature_previous_state,
    StateFeatureName.PreviousAction: _build_feature_previous_action_vectorized,
    StateFeatureName.Weather: _build_feature_weather,
    StateFeatureName.Zones: _build_feature_zones,
    StateFeatureName.CurrentZone: _build_feature_current_zone,
    StateFeatureName.SelectedZone: _build_feature_selected_zone,
    StateFeatureName.PreviousZoneAction: _build_feature_previous_zone_action,
    StateFeatureName.PreviousFullAction: _build_feature_previous_full_action
}


def _feature_node_features_size(nodes: List[Node], **kwargs) -> int:
    node_size = nodes[0].state_feature_size
    return node_size * len(nodes)


def _feature_truck_features_size(trucks: List[Truck], **kwargs) -> int:
    size = trucks[0].state_feature_size
    return size * len(trucks)


def _feature_node_size(nodes: List[Node], **kwargs) -> int:
    return len(nodes)


def _feature_truck_size(trucks: List[Truck], **kwargs) -> int:
    return len(trucks)


def _feature_current_truck_full_size(trucks: List[Truck], **kwargs) -> int:
    return trucks[0].state_feature_size


def _feature_cost_budget_size(**kwargs) -> int:
    return 1


def _feature_current_time_size(additional_properties: Dict[str, Any], **kwargs) -> int:
    if 'units_to_skip' not in additional_properties:
        raise AttributeError(f'State Feature {StateFeatureName.CurrentTime} requested '
                             f'but units_to_skip not present in additional_properties')
    units_to_skip = additional_properties['units_to_skip']
    size = 0
    for unit, max_value in STEP_UNITS_NORMALIZATION_FACTOR.items():
        if unit not in units_to_skip:
            size += max_value + 1
    return size


def _feature_dataset_time_size(additional_properties: Dict[str, Any], **kwargs) -> int:
    if 'units_to_skip' not in additional_properties:
        raise AttributeError(f'State Feature {StateFeatureName.DatasetTime} requested '
                             f'but units_to_skip not present in additional_properties')
    units_to_skip = additional_properties['units_to_skip']
    assert 'second_step' in units_to_skip and 'week' in units_to_skip, \
        f'{StateFeatureName.DatasetTime} feature does not support second_step and week unit'
    size = 0
    for unit, max_value in STEP_UNITS_NORMALIZATION_FACTOR.items():
        if unit not in units_to_skip:
            size += max_value + 1
    return size


def _feature_weather_size(weather_info: Dict[str, Any], **kwargs) -> int:
    return len(weather_info['conditions']) + 2


def _feature_zones_size(zones: Dict[str, Zone], **kwargs) -> int:
    return sum([zone.state_features_size for _, zone in zones.items()])


def _feature_current_zones_size(zones: Dict[str, Zone], **kwargs) -> int:
    for _, zone in zones.items():
        return zone.state_features_size


def _feature_previous_action_size(action_space: RepositionActionSpace, **kwargs) -> int:
    return action_space.target_space.size() + action_space.quantity_space.size()


def _feature_previous_zone_action_size(zone_action_space: ZoneActionSpace, **kwargs) -> int:
    return zone_action_space.zone_space.size()


def _feature_previous_full_action_size(
        action_space: RepositionActionSpace,
        zone_action_space: ZoneActionSpace,
        **kwargs
) -> int:
    return _feature_previous_action_size(action_space, **kwargs) + \
        _feature_previous_zone_action_size(zone_action_space, **kwargs)


def _feature_selected_zone_size(zones: Dict[str, Zone], **kwargs) -> int:
    return len(zones)


FEATURE_SIZE_MAPPING = {
    StateFeatureName.NodeFeatures: _feature_node_features_size,
    StateFeatureName.TruckFeatures: _feature_truck_features_size,
    StateFeatureName.CurrentTime: _feature_current_time_size,
    StateFeatureName.DatasetTime: _feature_dataset_time_size,
    StateFeatureName.TargetNode: _feature_node_size,
    StateFeatureName.CurrentTruck: _feature_truck_size,
    StateFeatureName.CurrentTruckFull: _feature_current_truck_full_size,
    StateFeatureName.PreviousState: lambda **kwargs: 0,
    StateFeatureName.PreviousAction: _feature_previous_action_size,
    StateFeatureName.Weather: _feature_weather_size,
    StateFeatureName.Zones: _feature_zones_size,
    StateFeatureName.CurrentZone: _feature_current_zones_size,
    StateFeatureName.SelectedZone: _feature_selected_zone_size,
    StateFeatureName.PreviousZoneAction: _feature_previous_zone_action_size,
    StateFeatureName.PreviousFullAction: _feature_previous_full_action_size,
}


class FeatureMissingError(Exception):

    def __init__(self, feature, all_features=tuple(FEATURE_BUILD_MAPPING.keys()), *args):
        message = f'State feature "{feature}" missing in features mapping: {all_features}'
        super(FeatureMissingError, self).__init__(message, *args)


def build_state(
        features: List[StateFeatureName],
        nodes: List[Node],
        trucks: List[Truck],
        additional_properties: Dict[str, Any],
        distances_provider: DistancesProvider,
        dtype=np.float,
        normalized=True,
        current_step: Optional[Step] = None,
        **kwargs
) -> State:
    state_features: List[StateFeatureValue] = []
    for feature in features:
        if feature in FEATURE_BUILD_MAPPING:
            feature_build_fn = FEATURE_BUILD_MAPPING[feature]
            state_features += feature_build_fn(nodes=nodes,
                                               trucks=trucks,
                                               normalized=normalized,
                                               additional_properties=additional_properties,
                                               current_step=current_step,
                                               distances_provider=distances_provider,
                                               **kwargs,
                                               )
        else:
            raise FeatureMissingError(feature)
    if len(state_features) > 0:
        return State(feature_values=tuple(state_features), dtype=dtype)
    else:
        raise Exception('State features array is empty. Impossible to build state')


def get_feature_size(
        feature: StateFeatureName,
        nodes: List[Node],
        trucks: List[Truck],
        additional_properties: Dict[str, Any],
        zones: Dict[str, Zone],
        action_space: RepositionActionSpace,
        zone_action_space: ZoneActionSpace,
        stacked_states: int = 1,
        weather_info: Dict[str, Any] = None,
        **kwargs,
) -> int:
    if feature in FEATURE_SIZE_MAPPING:
        size = FEATURE_SIZE_MAPPING[feature](nodes=nodes, trucks=trucks, additional_properties=additional_properties,
                                             weather_info=weather_info, zones=zones,
                                             action_space=action_space, zone_action_space=zone_action_space)
        if feature in STACKED_STATE_FEATURES:
            size *= stacked_states
        return size
    else:
        raise FeatureMissingError(feature)


def update_state_one_hot_encoding_feature(state: State, feature_name: Union[StateFeatureName, str],
                                          index: int, reset_value: int = 0):
    state.set_feature_value(feature_name, 1, index=index, reset_value=reset_value)


class StateBuilder:

    def __init__(
            self,
            config,
            nodes: List[Node],
            trucks: List[Truck],
            distances_provider: DistancesProvider,
            action_space: RepositionActionSpace,
            zone_action_space: Optional[ZoneActionSpace] = None,
            dtype=np.float,
            zones: Optional[Dict[str, Zone]] = None,
    ):
        self.config = config
        self.nodes: List[Node] = nodes
        self.trucks: List[Truck] = trucks
        self.zones: Dict[str, Zone] = zones
        self.dtype = dtype
        self.normalized: bool = self.config.state.normalized
        self.additional_properties: Dict[str, Any] = self.config.state.additional_properties
        self.distances_provider: DistancesProvider = distances_provider
        self.action_space: RepositionActionSpace = action_space
        self.zone_action_space: ZoneActionSpace = zone_action_space
        self.weather_info: Dict[str, Any] = self.load_weather_info()

        self.common_features: List[StateFeatureName] = self.config.state.common_features
        self.target_features: List[StateFeatureName] = self.config.state.target_features
        self.quantity_features: List[StateFeatureName] = self.config.state.quantity_features
        self.zone_features: List[StateFeatureName] = self.config.state.zone_features

        self.features_mapping: Dict[StateType, List[StateFeatureName]] = {
            StateType.Target: self.target_features,
            StateType.Quantity: self.quantity_features,
            StateType.Zone: self.zone_features,
        }
        self.current_time_step: Optional[Step] = None
        self.computed_size_features: Dict[StateFeatureName, int] = {}
        self.computed_features: Dict[StateFeatureName, List[StateFeatureValue]] = {}

        self.feature_size_mapping: Dict[str, int] = {
            'single_node_features': self.nodes[0].state_feature_size,
            'single_truck_features': self.trucks[0].state_feature_size,
            'single_zone_features': self.zones[list(self.zones.keys())[0]].state_features_size
        }

        self.states_no_previous_sizes: Dict[StateType, int] = {
            StateType.Target: 0,
            StateType.Quantity: 0,
        }

        self.previous_feature_keeper: Dict[StateType, Dict[int, Dict[StateFeatureName, Any]]] = {
            StateType.Target: {
                t.index: {
                    StateFeatureName.PreviousState: None,
                    StateFeatureName.PreviousAction: None,
                    StateFeatureName.PreviousZoneAction: None,
                } for t in self.trucks},
            StateType.Quantity: {
                t.index: {
                    StateFeatureName.PreviousState: None,
                    StateFeatureName.PreviousAction: None,
                    StateFeatureName.PreviousZoneAction: None,
                } for t in self.trucks},
            StateType.Zone: {
                t.index: {
                    StateFeatureName.PreviousState: None,
                    StateFeatureName.PreviousAction: None,
                    StateFeatureName.PreviousZoneAction: None,
                } for t in self.trucks}
        }

    def load_weather_info(self):
        with open(self.config.state.get_weather_info_path(), 'r') as f:
            return json.load(f)

    def set_previous_feature(self, state_type: StateType, truck: Truck, feature: StateFeatureName, value: Any):
        self.previous_feature_keeper[state_type][truck.index][feature] = value

    def get_previous_feature(self, state_type: StateType, truck: Truck, feature: StateFeatureName) -> Any:
        return self.previous_feature_keeper[state_type][truck.index][feature]

    def get_state_size(self, state_type: StateType) -> int:
        return self.feature_set_size(features=self.get_features_set(state_type), state_type=state_type)

    def get_state(
            self,
            state_type: StateType,
            current_step: Step,
            current_demand: StepData,
            current_zone_id: str = None,
            truck_index: int = None,
    ) -> State:
        return self.build_state(
            features=self.get_features_set(state_type),
            current_step=current_step,
            current_demand=current_demand,
            state_type=state_type,
            current_zone_id=current_zone_id,
            truck_index=truck_index,
        )

    def get_features_set(self, state_type: StateType) -> List[StateFeatureName]:
        if state_type == StateType.Zone:
            return self.features_mapping[state_type]
        return self.common_features + self.features_mapping[state_type]

    def get_all_features(self) -> List[StateFeatureName]:
        features: Dict[StateFeatureName, bool] = {}

        def add_features(feature_set: List[StateFeatureName]):
            for f in feature_set:
                if f not in features:
                    features[f] = True

        add_features(self.common_features)
        add_features(self.target_features)
        add_features(self.quantity_features)

        return list(features.keys())

    def get_node_distances(self, node_index) -> List[float]:
        return self.distances_provider.get_node_distances(self.nodes[node_index], nodes=self.nodes)

    def feature_set_size(self, features: List[StateFeatureName], state_type: StateType) -> int:
        set_size = 0
        previous_features = [StateFeatureName.PreviousState, StateFeatureName.PreviousAction]
        set_size_no_previous_features = 0
        for feature in features:
            # if feature in self.computed_size_features:
            #     size = self.computed_size_features[feature]
            #     set_size += size
            #     if feature not in previous_features:
            #         set_size_no_previous_features += size
            # else:
            feature_size = self.feature_size(feature)
            # self.computed_size_features[feature] = feature_size
            set_size += feature_size
            if feature not in previous_features:
                set_size_no_previous_features += feature_size

        self.states_no_previous_sizes[state_type] = set_size_no_previous_features
        if StateFeatureName.PreviousState in features:
            set_size += set_size_no_previous_features
        return set_size

    def feature_size(self, feature: StateFeatureName) -> int:
        return get_feature_size(feature, self.nodes, self.trucks, self.additional_properties,
                                weather_info=self.weather_info, zones=self.zones,
                                action_space=self.action_space, zone_action_space=self.zone_action_space)

    def build_state(
            self,
            features: List[StateFeatureName],
            current_step: Step,
            current_demand: StepData,
            state_type: StateType,
            **kwargs
    ) -> State:
        state_features: List[StateFeatureValue] = []
        # if self.current_time_step is None or self.current_time_step != current_step:
        #     self.computed_features: Dict[StateFeatureName, List[StateFeatureValue]] = {}
        #     self.current_time_step = current_step
        for feature in features:
            # if feature in self.computed_features:
            #     state_features += self.computed_features[feature]
            # else:
            state_size = self.states_no_previous_sizes[state_type]
            feature_values = self.build_state_feature_values(
                feature, current_step, current_demand, state_size=state_size, **kwargs)
            # if feature != StateFeatureName.PreviousState and feature != StateFeatureName.PreviousAction:
            #     self.computed_features[feature] = feature_values
            state_features += feature_values
        if len(state_features) > 0:
            return State(feature_values=tuple(state_features), dtype=self.dtype)
        else:
            raise Exception('State features array is empty. Impossible to build state')

    def build_state_feature_values(
            self,
            feature: StateFeatureName,
            current_step: Step,
            current_demand: StepData,
            **kwargs
    ) -> List[StateFeatureValue]:
        if feature in FEATURE_BUILD_MAPPING:
            feature_build_fn = FEATURE_BUILD_MAPPING[feature]
            return feature_build_fn(nodes=self.nodes,
                                    trucks=self.trucks,
                                    normalized=self.normalized,
                                    additional_properties=self.additional_properties,
                                    current_step=current_step,
                                    current_demand=current_demand,
                                    zones=self.zones,
                                    distances_provider=self.distances_provider,
                                    action_space=self.action_space,
                                    zone_action_space=self.zone_action_space,
                                    weather_info=self.weather_info,
                                    **kwargs,
                                    )
        else:
            raise FeatureMissingError(feature)

    def apply_action_on_state(
            self,
            state: State,
            step: Step,
            action: RepositionAction,
            truck: Truck,
            action_space: RepositionActionSpace,
    ):
        node_f_size = self.feature_size_mapping['single_node_features']
        truck_f_size = self.feature_size_mapping['single_truck_features']
        wait, node_i, quantity = action_space.action_to_action_value(action)
        if not wait:
            target_node = self.nodes[node_i]
            truck_new_features = truck.state_features(step, self.normalized)
            node_new_features = target_node.state_features(self.normalized)
            trucks_features = state.get_feature(name=StateFeatureName.TruckFeatures.value, values=True)[1]
            nodes_features = state.get_feature(name=StateFeatureName.NodeFeatures.value, values=True)[1]

            # set the new features for the truck that did the reposition
            trucks_features[truck.index * truck_f_size: truck.index * truck_f_size + truck_f_size] = truck_new_features
            state.set_feature_value(name=str(StateFeatureName.TruckFeatures.value), new_value=trucks_features.tolist())

            # set the new features for the node involved in the reposition
            nodes_features[node_i * node_f_size: node_i * node_f_size + node_f_size] = node_new_features
            state.set_feature_value(name=str(StateFeatureName.NodeFeatures.value), new_value=nodes_features.tolist())

    def apply_zone_action_on_state(
            self,
            state: State,
            step: Step,
            truck: Truck,
            previous_reposition_action: RepositionAction,
    ):
        r_wait, r_target, r_quantity = self.action_space.action_to_action_value(previous_reposition_action)
        truck_f_size = self.feature_size_mapping['single_truck_features']
        truck_new_features = truck.state_features(step, self.normalized)
        trucks_features = state.get_feature(name=StateFeatureName.TruckFeatures.value, values=True)[1]
        node_f_size = self.feature_size_mapping['single_node_features']
        node_new_features = truck.current_node.state_features(self.normalized)
        zones_features = state.get_feature(name=StateFeatureName.Zones.value, values=True)[1]
        zone_f_size = self.feature_size_mapping['single_zone_features']
        node_zone = truck.current_node.zone
        zone_feature_index = self.zone_action_space.zone_space.inverted_actions_mapping[node_zone]

        # set the new features for the truck that did the reposition
        trucks_features[truck.index * truck_f_size: truck.index * truck_f_size + truck_f_size] = truck_new_features
        state.set_feature_value(name=str(StateFeatureName.TruckFeatures.value), new_value=trucks_features.tolist())

        # set the new features for the node involved in the reposition
        # this is not correct, we need to slice by zone first and then by the node index inside its zone
        zone_slice = zones_features[zone_feature_index * zone_f_size: zone_feature_index * zone_f_size + zone_f_size]
        zone_slice[r_target * node_f_size: r_target * node_f_size + node_f_size] = node_new_features
        zones_features[zone_feature_index * zone_f_size: zone_feature_index * zone_f_size + zone_f_size] = zone_slice
        state.set_feature_value(name=str(StateFeatureName.Zones.value), new_value=zones_features.tolist())

    def apply_reposition_action_on_state(
            self,
            state: State,
            truck: Truck,
            current_zone_id: str,
            is_wait: bool,
            action: RepositionAction,
            action_space: RepositionActionSpace,
    ):
        wait, node_i, quantity = action_space.action_to_action_value(action)
        if not is_wait and truck.current_node.zone == current_zone_id:
            node_f_size = self.feature_size_mapping['single_node_features']
            node_new_features = truck.current_node.state_features(self.normalized)
            nodes_features = state.get_feature(name=StateFeatureName.NodeFeatures.value, values=True)[1]

            # set the new features for the node involved in the reposition
            nodes_features[node_i * node_f_size: node_i * node_f_size + node_f_size] = node_new_features
            state.set_feature_value(name=str(StateFeatureName.NodeFeatures.value), new_value=nodes_features.tolist())
