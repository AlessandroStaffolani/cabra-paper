from logging import Logger
from typing import List, Dict, Optional, Tuple

import numpy as np

from cabra import SingleRunConfig
from cabra.core.state import State
from cabra.core.step import Step
from cabra.environment.action_space import RepositionActionSpace
from cabra.environment.agent import AgentType
from cabra.environment.agent.abstract import SubActionUtils
from cabra.environment.agent.baseline import RandomAgent, BaselineAgent
from cabra.environment.data_structure import StateType, ActionType
from cabra.environment.node import Node, DistancesProvider
from cabra.environment.state_builder import StateBuilder
from cabra.environment.state_wrapper import StateWrapper
from cabra.environment.truck import Truck
from cabra.environment.zone import Zone


class ConstrainedSubActionUtils(SubActionUtils):

    def __init__(
            self,
            action_space: RepositionActionSpace,
            nodes: List[Node],
            zones: Dict[str, Zone],
            config: SingleRunConfig,
            nodes_max_distance: float,
            state_builder: StateBuilder,
            critical_threshold: float,
            max_distance: float,
            zone_max_distance: float,
            zones_filtered_size: int,
            distances_provider: DistancesProvider
    ):
        super().__init__(action_space, nodes, zones, config, nodes_max_distance, state_builder)
        self.critical_threshold: float = critical_threshold
        self.max_distance: float = max_distance
        self.zone_max_distance: float = zone_max_distance
        self.distances_provider: DistancesProvider = distances_provider
        self.zones_filtered_size: int = zones_filtered_size

        self.next_is_pick_mapping: Dict[int, bool] = {i: True for i in range(self.config.environment.trucks.n_trucks)}
        self.enforce_next_pick: Optional[bool] = None

        self.current_full_zone_metric = {}
        self.current_empty_zone_metric = {}

    def get_zone_overloaded_stats(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        # for each zone, it contains the number of nodes that are under critic threshold of empty slots -> full nodes
        full_zone_metric = {}
        # for each zone, it contains the number of nodes that are under critic threshold of bikes -> empty nodes
        empty_zone_metric = {}
        for zone_id, zone in self.zones.items():
            for node in zone.nodes:
                if node.is_full_critical(self.critical_threshold):
                    # this node is going to be full
                    if zone_id not in full_zone_metric:
                        full_zone_metric[zone_id] = 1
                    else:
                        full_zone_metric[zone_id] += 1
                if node.is_empty_critical(self.critical_threshold):
                    # this node is going to be empty
                    if zone_id not in empty_zone_metric:
                        empty_zone_metric[zone_id] = 1
                    else:
                        empty_zone_metric[zone_id] += 1
        return full_zone_metric, empty_zone_metric

    def prepare_zone_action_space(self, truck: Truck):
        # for each zone, it contains the number of nodes that are under critic threshold of empty slots -> full nodes
        self.current_full_zone_metric, self.current_empty_zone_metric = self.get_zone_overloaded_stats()
        next_is_pick = self.next_is_pick_mapping[truck.index]
        current_zone = self.zones[truck.current_node.zone]
        # filter out zones that are too distant from current
        full_zone_metric = {
            zone_id: full_nodes for zone_id, full_nodes in self.current_full_zone_metric.items()
            if current_zone.distance(self.zones[zone_id].centroid) < self.zone_max_distance}
        empty_zone_metric = {
            zone_id: empty_nodes for zone_id, empty_nodes in self.current_empty_zone_metric.items()
            if current_zone.distance(self.zones[zone_id].centroid) < self.zone_max_distance}
        filtered_zones_set: List[str] = []
        if next_is_pick:
            # pick action -> search for the zones_filtered_size fullest zones, with at least one full node
            # that are less distant than zone_max_distance, then randomly pick one
            key_reversed_set: Dict[int, str] = {
                full_nodes: zone_id for zone_id, full_nodes in full_zone_metric.items() if full_nodes > 0
            }
            for full_nodes in sorted(key_reversed_set.keys()):
                filtered_zones_set.append(key_reversed_set[full_nodes])
        else:
            # drop action -> search for the zones_filtered_size emptiest zones, with at least one empty node
            # that are less distant than zone_max_distance, then randomly pick one
            key_reversed_set: Dict[int, str] = {
                empty_nodes: zone_id for zone_id, empty_nodes in empty_zone_metric.items()
                if empty_nodes > 0
            }
            for empty_nodes in sorted(key_reversed_set.keys(), reverse=True):
                filtered_zones_set.append(key_reversed_set[empty_nodes])

        if len(filtered_zones_set) == 0:
            # all zones are ok now, we can only wait
            self.action_space.zone_space.disable_all_except_wait()
        else:
            if len(filtered_zones_set) > self.zones_filtered_size:
                filtered_zones_set = filtered_zones_set[:self.zones_filtered_size]
            for zone_action in self.action_space.zone_space.get_available_actions():
                zone_id = self.action_space.zone_space.actions_mapping[zone_action]
                if self.action_space.zone_space.is_wait_action(zone_action):
                    self.action_space.zone_space.disable_action(zone_action)
                elif zone_id in filtered_zones_set:
                    self.action_space.zone_space.enable_action(zone_action)
                else:
                    self.action_space.zone_space.disable_action(zone_action)
        # swap next_is_pick
        self.next_is_pick_mapping[truck.index] = not next_is_pick

    def prepare_target_node_sub_action(self, truck: Truck, current_zone_id: Optional[str]):
        all_disabled = True
        next_is_pick = self.next_is_pick_mapping[truck.index]
        self.enforce_next_pick = next_is_pick
        stats = {'critically_empty_nodes': 0, 'critically_full_nodes': 0, 'nodes_fullness_ratio': {}}
        if current_zone_id is not None:
            # we can disable actions only if current_zone_id is not None,
            # otherwise it means zone_action is wait, and we can only. Only wait is already set by the environment
            zone_nodes = self.zones[current_zone_id].nodes
            for node_index, node in enumerate(zone_nodes):
                action_index = self.action_space.target_space.inverted_actions_mapping[node_index]
                distance = self.distances_provider.get_nodes_distance(truck.current_node, node)
                if distance < self.max_distance:
                    if next_is_pick:
                        if node.is_full_critical(self.critical_threshold):
                            # node has many bikes, enable this
                            self.action_space.target_space.enable_action(action_index)
                            all_disabled = False
                            stats['critically_full_nodes'] += 1
                        else:
                            # node has more empty slots than threshold, disable this
                            self.action_space.target_space.disable_action(action_index)
                    else:
                        if node.is_empty_critical(self.critical_threshold):
                            # node has few bikes, enable this
                            self.action_space.target_space.enable_action(action_index)
                            all_disabled = False
                            stats['critically_empty_nodes'] += 1
                        else:
                            # node has more bikes than threshold, disable this
                            self.action_space.target_space.disable_action(action_index)
                    stats['nodes_fullness_ratio'][node.fullness_ratio()] = node_index
                else:
                    # node too much distant, we ignore it by default
                    self.action_space.target_space.disable_action(action_index)
                    # empty to full swap required?
            # if truck.load == 0 and stats['critically_empty_nodes'] > 0 and stats['critically_full_nodes'] == 0:
            #     self.empty_to_full_rule_swap(stats['critically_empty_nodes'], stats['nodes_fullness_ratio'],
            #                                  truck)
            # elif truck.empty_slots == 0 and stats['critically_full_nodes'] > 0 and stats['critically_empty_nodes'] == 0:
            #     # full to empty swap required?
            #     self.full_to_empty_rule_swap(stats['critically_full_nodes'], stats['nodes_fullness_ratio'],
            #                                  truck)
            if all_disabled:
                # all nodes are disable, we ensure wait is enabled
                self.action_space.target_space.enable_wait_action()
            else:
                # at least one node is not disabled, we disable wait
                self.action_space.target_space.disable_wait_action()

    def empty_to_full_rule_swap(self, nodes_to_enable: int, nodes_fullness_ratio: Dict[float, int], truck: Truck):
        self.action_space.target_space.disable_all()
        indexes_to_enable = [
            nodes_fullness_ratio[key] for key in sorted(nodes_fullness_ratio.keys(), reverse=True)[:nodes_to_enable]]
        for node_index in indexes_to_enable:
            self.action_space.target_space.enable_action(node_index)
        # quantity sub-action must be a pick now
        self.next_is_pick_mapping[truck.index] = True
        self.enforce_next_pick = True

    def full_to_empty_rule_swap(self, nodes_to_enable: int, nodes_fullness_ratio: Dict[float, int], truck: Truck):
        self.action_space.target_space.disable_all()
        indexes_to_enable = [
            nodes_fullness_ratio[key] for key in sorted(nodes_fullness_ratio.keys(), reverse=False)[:nodes_to_enable]]
        for node_index in indexes_to_enable:
            self.action_space.target_space.enable_action(node_index)
        # quantity sub-action must be a drop now
        self.next_is_pick_mapping[truck.index] = False
        self.enforce_next_pick = False

    def prepare_quantity_sub_action(
            self,
            target_node_action: int,
            truck: Truck,
            state_wrapper: StateWrapper,
            current_zone_id: Optional[str]
    ):
        next_is_pick = self.next_is_pick_mapping[truck.index]
        if current_zone_id is not None:
            # we can disable actions only if current_zone_id is not None,
            # otherwise it means zone_action is wait, and we can only. Only wait is already set by the environment
            zone_nodes = self.zones[current_zone_id].nodes
            if not self.action_space.target_space.is_wait_action(target_node_action):
                # populate the TargetNode feature
                # state_wrapper.set_target_node(target_node_action)
                target_node: Node = zone_nodes[target_node_action]

                for q_action in self.action_space.quantity_space.get_available_actions():
                    if not self.action_space.quantity_space.is_wait_action(q_action):
                        q_value = self.action_space.quantity_space.actions_mapping[q_action]
                        if next_is_pick:
                            # is pick action
                            if q_value > 0:
                                # pick action
                                if not truck.pick_possible(target_node, q_value):
                                    self.action_space.quantity_space.disable_action(q_action)
                                else:
                                    self.action_space.quantity_space.enable_action(q_action)
                            else:
                                # it is a pick action, all the drop actions are disabled
                                self.action_space.quantity_space.disable_action(q_action)
                        else:
                            # is drop action
                            if q_value < 0:
                                # drop action
                                if not truck.drop_possible(target_node, q_value):
                                    self.action_space.quantity_space.disable_action(q_action)
                                else:
                                    self.action_space.quantity_space.enable_action(q_action)
                            else:
                                # it is a drop action, all the pick actions are disabled
                                self.action_space.quantity_space.disable_action(q_action)
                if len(self.action_space.quantity_space.get_available_actions()) == 1:
                    self.action_space.quantity_space.enable_wait_action()
                else:
                    self.action_space.quantity_space.disable_wait_action()
            else:
                # is wait action, we can only wait
                self.action_space.quantity_space.disable_all_except_wait()
            if len(self.action_space.quantity_space.get_available_actions()) == 0:
                self.action_space.quantity_space.enable_wait_action()
        # we swap this variable
        self.next_is_pick_mapping[truck.index] = not next_is_pick


class ConstrainedRandomAgent(RandomAgent):
    """
    Random Agent. It randomly selects actions among the available
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
        super(ConstrainedRandomAgent, self).__init__(action_space=action_space,
                                                     random_state=random_state,
                                                     action_spaces=action_spaces,
                                                     state_spaces=state_spaces,
                                                     nodes=nodes,
                                                     zones=zones,
                                                     nodes_max_distance=nodes_max_distance,
                                                     log=log,
                                                     state_builder=state_builder,
                                                     config=config,
                                                     **kwargs)
        self.critical_threshold: float = self.config.environment.constrained_space.critical_threshold
        self.max_distance: float = self.config.environment.constrained_space.max_distance
        self.zone_max_distance: float = self.config.environment.constrained_space.zone_max_distance
        self.zones_filtered_size: int = self.config.environment.constrained_space.zones_filtered_size
        self.sub_actions_utils: ConstrainedSubActionUtils = ConstrainedSubActionUtils(
            action_space=self.action_space,
            config=self.config,
            nodes=self.nodes,
            zones=zones,
            nodes_max_distance=nodes_max_distance,
            state_builder=self.state_builder,
            critical_threshold=self.critical_threshold,
            max_distance=self.max_distance,
            zone_max_distance=self.zone_max_distance,
            zones_filtered_size=self.zones_filtered_size,
            distances_provider=distances_provider,
        )
        self.name = AgentType.ConstrainedRandom
        self.zones_next_is_pick_mapping: Dict[int, bool] = {
            i: True for i in range(self.config.environment.trucks.n_trucks)}

    # def choose_zone_action(self, state: State, t: Step, epsilon: float, random: float, truck: Truck) -> int:
    #     full_zone_metric, empty_zone_metric = self.sub_actions_utils.get_zone_overloaded_stats()
    #     next_is_pick = self.zones_next_is_pick_mapping[truck.index]
    #     current_zone = self.zones[truck.current_node.zone]
    #     # filter out zones that are too distance from current
    #     full_zone_metric = {
    #         zone_id: full_nodes for zone_id, full_nodes in full_zone_metric.items()
    #         if current_zone.distance(self.zones[zone_id].centroid) < self.zone_max_distance}
    #     empty_zone_metric = {
    #         zone_id: empty_nodes for zone_id, empty_nodes in empty_zone_metric.items()
    #         if current_zone.distance(self.zones[zone_id].centroid) < self.zone_max_distance}
    #     filtered_zones_set: List[str] = []
    #     if next_is_pick:
    #         # pick action -> search for the zones_filtered_size fullest zones, with at least one full node
    #         # that are less distant than zone_max_distance, then randomly pick one
    #         key_reversed_set: Dict[int, str] = {
    #             full_nodes: zone_id for zone_id, full_nodes in full_zone_metric.items() if full_nodes > 0
    #         }
    #         for full_nodes in sorted(key_reversed_set.keys()):
    #             filtered_zones_set.append(key_reversed_set[full_nodes])
    #     else:
    #         # drop action -> search for the zones_filtered_size emptiest zones, with at least one empty node
    #         # that are less distant than zone_max_distance, then randomly pick one
    #         key_reversed_set: Dict[int, str] = {
    #             empty_nodes: zone_id for zone_id, empty_nodes in empty_zone_metric.items()
    #             if empty_nodes > 0
    #         }
    #         for empty_nodes in sorted(key_reversed_set.keys(), reverse=True):
    #             filtered_zones_set.append(key_reversed_set[empty_nodes])
    #     # toggle next_is_pick
    #     self.zones_next_is_pick_mapping[truck.index] = not next_is_pick
    #     if len(filtered_zones_set) == 0:
    #         # all zones are ok now
    #         return self.action_space.zone_space.wait_action_index
    #     else:
    #         if len(filtered_zones_set) > self.zones_filtered_size:
    #             filtered_zones_set = filtered_zones_set[:self.zones_filtered_size]
    #         random_zone = self.random.choice(filtered_zones_set)
    #         zone_action_index = self.action_space.zone_space.inverted_actions_mapping[random_zone]
    #         return zone_action_index


class ConstrainedGreedyAgent(BaselineAgent):

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
        super(ConstrainedGreedyAgent, self).__init__(action_space=action_space,
                                                     random_state=random_state,
                                                     name=AgentType.ConstrainedGreedy,
                                                     action_spaces=action_spaces,
                                                     state_spaces=state_spaces,
                                                     nodes=nodes,
                                                     zones=zones,
                                                     nodes_max_distance=nodes_max_distance,
                                                     log=log,
                                                     state_builder=state_builder,
                                                     config=config,
                                                     **kwargs)
        self.critical_threshold: float = self.config.environment.constrained_space.critical_threshold
        self.max_distance: float = self.config.environment.constrained_space.max_distance
        self.zone_max_distance: float = self.config.environment.constrained_space.zone_max_distance
        self.zones_filtered_size: int = self.config.environment.constrained_space.zones_filtered_size
        self.sub_actions_utils: ConstrainedSubActionUtils = ConstrainedSubActionUtils(
            action_space=self.action_space,
            config=self.config,
            nodes=self.nodes,
            zones=zones,
            nodes_max_distance=nodes_max_distance,
            state_builder=self.state_builder,
            critical_threshold=self.critical_threshold,
            max_distance=self.max_distance,
            zone_max_distance=self.zone_max_distance,
            zones_filtered_size=self.zones_filtered_size,
            distances_provider=distances_provider,
        )
        self.next_is_pick_mapping: Dict[int, bool] = {i: True for i in range(self.config.environment.trucks.n_trucks)}
        self.zones_next_is_pick_mapping: Dict[int, bool] = {
            i: True for i in range(self.config.environment.trucks.n_trucks)}

    def choose_target_node(
            self,
            state: State,
            t: Step,
            epsilon: float,
            random: float,
            truck: Truck,
            current_zone_id: str,
    ) -> int:
        actions_output = []
        available_actions = self.action_space.target_space.get_available_actions()
        next_is_pick = self.sub_actions_utils.enforce_next_pick
        test_value = None
        for action_index in available_actions:
            if not self.action_space.target_space.is_wait_action(action_index):
                node_index = self.action_space.target_space.actions_mapping[action_index]
                node = self.zones[current_zone_id].nodes[node_index]
                if next_is_pick:
                    # is pick action
                    if test_value is None or node.empty_slots < test_value:
                        test_value = node.empty_slots
                        actions_output = [action_index]
                    elif node.empty_slots == test_value:
                        actions_output.append(action_index)
                else:
                    # is drop action
                    if test_value is None or node.bikes < test_value:
                        test_value = node.bikes
                        actions_output = [action_index]
                    elif node.bikes == test_value:
                        actions_output.append(action_index)
        if len(actions_output) == 0:
            return self.action_space.target_space.wait_action_index
        else:
            return self.random.choice(actions_output)

    def choose_quantity(
            self,
            state: State,
            t: Step,
            epsilon: float,
            random: float,
            truck: Truck,
            current_zone_id: str,
    ) -> int:
        next_is_pick = self.sub_actions_utils.enforce_next_pick
        available_actions = self.action_space.quantity_space.get_available_actions()

        action_output = self.action_space.quantity_space.wait_action_index
        max_value = 0
        for action_index in available_actions:
            if not self.action_space.quantity_space.is_wait_action(action_index):
                quantity = self.action_space.quantity_space.actions_mapping[action_index]
                if abs(quantity) > max_value:
                    max_value = abs(quantity)
                    action_output = action_index

        self.next_is_pick_mapping[truck.index] = not next_is_pick
        return action_output

    def choose_zone_action(self, state: State, t: Step, epsilon: float, random: float, truck: Truck) -> int:
        next_is_pick = self.zones_next_is_pick_mapping[truck.index]
        current_zone = self.zones[truck.current_node.zone]
        # filter out zones that are too distant from current
        full_zone_metric = {
            zone_id: full_nodes for zone_id, full_nodes in self.sub_actions_utils.current_full_zone_metric.items()
            if current_zone.distance(self.zones[zone_id].centroid) < self.zone_max_distance}
        empty_zone_metric = {
            zone_id: empty_nodes for zone_id, empty_nodes in self.sub_actions_utils.current_empty_zone_metric.items()
            if current_zone.distance(self.zones[zone_id].centroid) < self.zone_max_distance}
        action_zone = None
        max_value = None
        if next_is_pick:
            # pick action -> search for the zones_filtered_size fullest zones, with at least one full node
            # that are less distant than zone_max_distance, then randomly pick one
            for zone_id, full_nodes in full_zone_metric.items():
                if full_nodes > 0 and (max_value is None or full_nodes > max_value):
                    max_value = full_nodes
                    action_zone = zone_id
        else:
            # drop action -> search for the zones_filtered_size emptiest zones, with at least one empty node
            # that are less distant than zone_max_distance, then randomly pick one
            for zone_id, empty_nodes in empty_zone_metric.items():
                if empty_nodes > 0 and (max_value is None or empty_nodes > max_value):
                    max_value = empty_nodes
                    action_zone = zone_id
        # toggle next_is_pick
        self.zones_next_is_pick_mapping[truck.index] = not next_is_pick
        if action_zone is None:
            # all zones are ok now
            return self.action_space.zone_space.wait_action_index
        else:
            zone_action_index = self.action_space.zone_space.inverted_actions_mapping[action_zone]
            return zone_action_index
