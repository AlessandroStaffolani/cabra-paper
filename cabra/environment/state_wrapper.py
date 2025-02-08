from typing import List, Dict, Union, Optional

from cabra.core.state import State
from cabra.core.step import Step
from cabra.core.step_data import StepData
from cabra.environment.data_structure import StateFeatureName, StateType
from cabra.environment.state_builder import StateBuilder, update_state_one_hot_encoding_feature
from cabra.environment.truck import Truck


class StateWrapper:

    def __init__(
            self,
            state_types: List[StateType],
            current_step: Step,
            current_demand: StepData,
            state_builder: Optional[StateBuilder] = None,
            states: Optional[Dict[StateType, State]] = None,
            current_zone_id: Optional[str] = None,
            current_truck_index: Optional[int] = None,
    ):
        assert state_builder is not None or states is not None, \
            'StateWrapper requires state_builder or states arguments to not be None'
        self.state_types: List[StateType] = state_types
        self.current_step: Step = current_step
        self.current_demand: StepData = current_demand
        self.current_zone_id: Optional[str] = current_zone_id
        self.current_truck_index: Optional[int] = current_truck_index
        if states is None:
            self.states: Dict[StateType, State] = {
                state_type: state_builder.get_state(
                    state_type, self.current_step,
                    self.current_demand, self.current_zone_id,
                    self.current_truck_index) for state_type in self.state_types
            }
        else:
            self.states: Dict[StateType, State] = states

    def get_state(self, state_type: Union[StateType, str]) -> State:
        if isinstance(state_type, str):
            state_type = StateType(state_type)
        return self.states[state_type]

    def update_state_one_hot_encoding_feature(
            self,
            state_type: StateType,
            index: int,
            feature_name: StateFeatureName,
            reset_value: int = 0
    ):
        update_state_one_hot_encoding_feature(
            state=self.states[state_type],
            feature_name=feature_name,
            index=index,
            reset_value=reset_value)

    def update_feature_value(self, feature_name, new_value):
        for _, state in self.states.items():
            state.set_feature_value(feature_name, new_value)

    def set_current_truck(self, truck: Truck):
        for state_type, state in self.states.items():
            if StateFeatureName.CurrentTruck.value in state.features_names():
                self.update_state_one_hot_encoding_feature(state_type, truck.index, StateFeatureName.CurrentTruck, 0)

    def set_target_node(self, node_index: int, state_type: StateType = StateType.Quantity):
        self.update_state_one_hot_encoding_feature(state_type, node_index, StateFeatureName.TargetNode, 0)

    def set_previous_state(self, state_builder: StateBuilder, truck: Truck):
        for state_type, state in self.states.items():
            previous_state = state_builder.get_previous_feature(state_type, truck, StateFeatureName.PreviousState)
            if previous_state is not None:
                state.set_feature_value(StateFeatureName.PreviousState.value, previous_state.tolist())

    def set_previous_action(self, state_builder: StateBuilder, truck: Truck):
        for state_type, state in self.states.items():
            previous_action = state_builder.get_previous_feature(state_type, truck, StateFeatureName.PreviousAction)
            if previous_action is not None:
                target_values = [0] * state_builder.action_space.target_space.size()
                quantity_values = [0] * state_builder.action_space.quantity_space.size()
                target_values[previous_action.target] = 1
                quantity_values[previous_action.quantity] = 1
                state.set_feature_value(StateFeatureName.PreviousAction.value, target_values + quantity_values)

    def set_previous_zone_action(self, state_builder: StateBuilder, truck: Truck):
        for state_type, state in self.states.items():
            previous_action = state_builder.get_previous_feature(state_type, truck, StateFeatureName.PreviousZoneAction)
            if previous_action is not None:
                zone_values = [0] * state_builder.zone_action_space.zone_space.size()
                zone_values[previous_action.zone] = 1
                state.set_feature_value(StateFeatureName.PreviousZoneAction.value, zone_values)

    def set_previous_full_action(self, state_builder: StateBuilder, truck: Truck):
        for state_type, state in self.states.items():
            previous_action = state_builder.get_previous_feature(state_type, truck, StateFeatureName.PreviousAction)
            previous_zone_action = state_builder.get_previous_feature(
                state_type, truck, StateFeatureName.PreviousZoneAction)
            if previous_zone_action is not None and previous_action is not None:
                zone_values = [0] * state_builder.zone_action_space.zone_space.size()
                zone_values[previous_zone_action.zone] = 1
                target_values = [0] * state_builder.action_space.target_space.size()
                quantity_values = [0] * state_builder.action_space.quantity_space.size()
                target_values[previous_action.target] = 1
                quantity_values[previous_action.quantity] = 1
                values = zone_values + target_values + quantity_values
                state.set_feature_value(StateFeatureName.PreviousFullAction.value, values)

    def __getitem__(self, item: Union[int, StateType]) -> State:
        if isinstance(item, int):
            key = self.state_types[item]
        else:
            key = item
        return self.get_state(key)

    def to_dict(self) -> Dict[Union[StateType, str], Union[State, Step]]:
        d: Dict[Union[StateType, str], Union[State, Step]] = self.states
        d['step'] = self.current_step
        return d

    def items(self):
        return self.states.items()

    def copy(self) -> 'StateWrapper':
        states_copy = {s_type: s.copy() for s_type, s in self.states.items()}
        return StateWrapper(
            state_types=self.state_types,
            current_step=self.current_step,
            current_demand=self.current_demand,
            state_builder=None,
            states=states_copy
        )
