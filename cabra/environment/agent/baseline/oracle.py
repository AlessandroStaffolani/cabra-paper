from logging import Logger
from typing import List, Dict, Optional, Tuple, Union, Any

import numpy as np

from cabra import SingleRunConfig
from cabra.core.step import Step
from cabra.emulator.models import RealDataModel
from cabra.environment.action_space import RepositionActionSpace, RepositionAction, ZoneAction
from cabra.environment.agent import AgentType
from cabra.environment.agent.baseline import ConstrainedRandomAgent
from cabra.environment.agent.baseline.constrained import ConstrainedSubActionUtils
from cabra.environment.data_structure import StateType, ActionType
from cabra.environment.node import Node, DistancesProvider
from cabra.environment.state_builder import StateBuilder
from cabra.environment.state_wrapper import StateWrapper
from cabra.environment.truck import Truck
from cabra.environment.zone import Zone


class NStepsOracleAgent(ConstrainedRandomAgent):
    """
    Oracle that sees the demand ahead of n steps. it selects the action using the ConstrainedRandomAgent policy
    """

    def __init__(
            self,
            action_space: RepositionActionSpace,
            random_state: np.random.RandomState,
            action_spaces: Dict[ActionType, int],
            state_spaces: Dict[StateType, int],
            nodes: List[Node],
            zones: Dict[str, Zone],
            nodes_max_distance: float,
            log: Logger,
            state_builder: StateBuilder,
            distances_provider: DistancesProvider,
            config: Optional[SingleRunConfig] = None,
            **kwargs
    ):
        super().__init__(action_space, random_state, action_spaces, state_spaces, nodes, zones, nodes_max_distance,
                         log, state_builder, distances_provider, config, **kwargs)
        self.name = AgentType.NStepsOracle
        self.look_ahead_steps: int = self.config.environment.agent.n_steps_oracle.look_ahead_steps
        self.nodes_copy: List[Node] = [
            Node(node.name, node.index, node.position, node.bikes, node.total_slots, node.node_features,
                 node.zone, node.shortage_threshold, node.critical_threshold, node.critical_normalized)
            for node in self.nodes
        ]
        self.nodes_copy_mapping: Dict[str, Node] = {n.name: n for n in self.nodes_copy}
        self.zones_copy: Dict[str, Zone] = self.get_zones_copy()

        self.sub_actions_utils_prediction: ConstrainedSubActionUtils = ConstrainedSubActionUtils(
            action_space=self.action_space,
            config=self.config,
            nodes=self.nodes_copy,
            zones=self.zones_copy,
            nodes_max_distance=nodes_max_distance,
            state_builder=self.state_builder,
            critical_threshold=self.critical_threshold,
            max_distance=self.max_distance,
            zone_max_distance=self.zone_max_distance,
            zones_filtered_size=self.zones_filtered_size,
            distances_provider=distances_provider,
        )
        self.load_generator: Optional[RealDataModel] = None
        self.step_size: int = self.config.environment.time_step.step_size
        self.should_apply_action: bool = True
        self.predicted_actions: int = 0
        self.not_predicted_actions: int = 0

    def get_zones_copy(self) -> Dict[str, Zone]:
        zones: Dict[str, Zone] = {}
        for node in self.nodes_copy:
            if node.zone not in zones:
                centroid = self.zones[node.zone].centroid
                zones[node.zone] = Zone(node.zone, [node], centroid)
                zones[node.zone].max_zone_size = self.zones[node.zone].max_zone_size
            else:
                zones[node.zone].add_node(node)
        return zones

    def choose(
            self,
            state_wrapper: StateWrapper,
            t: Step,
            truck: Truck,
            current_zone_id: Optional[str] = None
    ) -> Tuple[Union[RepositionAction, ZoneAction], Optional[Dict[str, Any]]]:
        self.forward_nodes_demand(t)
        action, action_info = super().choose(state_wrapper, t, truck, current_zone_id)
        self.apply_action_on_nodes_copy(action, current_zone_id)
        return action, action_info

    def _choose_action(
            self,
            state_wrapper: StateWrapper,
            t: Step,
            truck: Truck,
            current_zone_id: str,
            epsilon: float,
            random: float,
            **kwargs
    ) -> RepositionAction:
        # prepare target_node sub-action, using prediction data
        self.sub_actions_utils_prediction.prepare_target_node_sub_action(truck, current_zone_id)
        # choose the target node
        t_node_a = self.choose_target_node(
            state_wrapper.get_state(StateType.Target), t, epsilon, random, truck, current_zone_id)
        # prepare the quantity sub-action, using prediction data
        self.sub_actions_utils_prediction.prepare_quantity_sub_action(t_node_a, truck, state_wrapper, current_zone_id)
        # chose quantity
        q_a = self.choose_quantity(
            state_wrapper.get_state(StateType.Target), t, epsilon, random, truck, current_zone_id)

        action = RepositionAction(target=t_node_a, quantity=q_a)

        if not self.is_reposition_valid(action, current_zone_id):
            self.should_apply_action = False
            return super()._choose_action(state_wrapper, t, truck, current_zone_id, epsilon, random)

        self.should_apply_action = True
        return action

    def _choose_zone(
            self,
            state_wrapper: StateWrapper,
            t: Step,
            truck: Truck,
            epsilon: float,
            random: float,
            **kwargs
    ) -> ZoneAction:
        # prepare zone action space, using prediction data
        self.sub_actions_utils_prediction.prepare_zone_action_space(truck)
        zone_a = self.choose_zone_action(state_wrapper.get_state(StateType.Zone), t, epsilon, random, truck)
        return ZoneAction(zone=zone_a)

    def forward_nodes_demand(self, current_t: Step):
        # get next look_ahead_steps demand and update nodes copy
        oracle_step = self._forward_time_step(current_t)
        oracle_model_index = self._get_oracle_model_index(self.look_ahead_steps)
        oracle_demand = self.load_generator.get_step_data_internal(oracle_model_index, oracle_step, True, True)
        self.load_generator.update_nodes_capacity(oracle_demand, False, nodes_mapping=self.nodes_copy_mapping)

    def apply_action_on_nodes_copy(self, action: Union[RepositionAction, ZoneAction], current_zone_id: str):
        if not self.is_zone_agent and self.should_apply_action:
            # apply action
            wait, target_index, quantity_val = self.action_space.action_to_action_value(action)
            if not wait:
                target_node = self.zones_copy[current_zone_id].nodes[target_index]
                if quantity_val < 0:
                    # drop action -> allocate quantity_val to target node
                    target_node.allocate(abs(quantity_val))
                else:
                    # pick action -> remove quantity_val node
                    target_node.remove(quantity_val)
            self.predicted_actions += 1
        elif not self.is_zone_agent and not self.should_apply_action:
            self.not_predicted_actions += 1

    def is_reposition_valid(self, action: RepositionAction, current_zone_id: str) -> bool:
        # check if action is valid in the current step (since we select it from the future)
        wait, target_index, quantity_val = self.action_space.action_to_action_value(action)
        valid = True
        if not wait:
            target_node = self.zones[current_zone_id].nodes[target_index]
            if quantity_val < 0:
                # drop action -> allocate quantity_val to target node
                valid = target_node.allocation_possible(abs(quantity_val))
            else:
                # pick action -> remove quantity_val node
                valid = target_node.removal_possible(quantity_val)
        return valid

    def _forward_time_step(self, current_t: Step) -> Step:
        return current_t.add(Step.from_total_steps(self.step_size * self.look_ahead_steps))

    def _get_oracle_model_index(self, forward_n: int) -> int:
        oracle_model_index = self.load_generator.current_index + forward_n
        if str(oracle_model_index) not in self.load_generator.dataset:
            return self._get_oracle_model_index(forward_n - 1)
        else:
            return oracle_model_index
