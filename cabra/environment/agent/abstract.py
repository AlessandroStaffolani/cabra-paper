import pickle
from abc import abstractmethod
from collections import namedtuple
from logging import Logger
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
from torch import Tensor

from cabra import SingleRunConfig, single_run_config
from cabra.common.data_structure import RunMode
from cabra.common.stats_tracker import Tracker
from cabra.core.state import State
from cabra.core.step import Step
from cabra.emulator.models import AbstractModel
from cabra.environment.action_space import RepositionAction, RepositionActionSpace, ZoneActionSpace, \
    Action, ZoneAction
from cabra.environment.agent import AgentType
from cabra.environment.agent.experience_replay import RolloutBuffer
from cabra.environment.data_structure import StateType, ActionType
from cabra.environment.node import Node
from cabra.environment.state_builder import StateBuilder
from cabra.environment.state_wrapper import StateWrapper
from cabra.environment.truck import Truck
from cabra.environment.zone import Zone

AgentLoss = namedtuple('AgentLoss', 'add_node_actor add_node_critic combined_actor combined_critic')


class AgentLearnMetric:

    def __init__(self, state_type: StateType, name: str, value: float):
        self.state_type: StateType = state_type
        self.name: str = name
        self.value: float = value


class SubActionUtils:

    def __init__(self,
                 action_space: Union[RepositionActionSpace, ZoneActionSpace],
                 nodes: List[Node],
                 zones: Dict[str, Zone],
                 config: SingleRunConfig,
                 nodes_max_distance: float,
                 state_builder: StateBuilder,
                 ):
        self.config: SingleRunConfig = config
        self.min_quantity: int = self.config.environment.action_space.min_quantity
        self.nodes: List[Node] = nodes
        self.zones: Dict[str, Zone] = zones
        self.nodes_max_distance: float = nodes_max_distance
        self.action_space: Union[RepositionActionSpace, ZoneActionSpace] = action_space
        self.state_builder: StateBuilder = state_builder

    def prepare_zone_action_space(self, truck: Truck):
        pass

    def prepare_target_node_sub_action(self, truck: Truck, current_zone_id: Optional[str]):
        if current_zone_id is not None:
            # we can disable actions only if current_zone_id is not None,
            # otherwise it means zone_action is wait, and we can only. Only wait is already set by the environment
            zone_nodes = self.zones[current_zone_id].nodes
            if truck.empty_slots == 0:
                # if truck is full we disable all full nodes
                for t_action in self.action_space.target_space.get_available_actions():
                    if not self.action_space.target_space.is_wait_action(t_action):
                        t_index = self.action_space.target_space.actions_mapping[t_action]
                        t_node = zone_nodes[t_index]
                        if t_node.empty_slots < self.min_quantity:
                            self.action_space.target_space.disable_action(t_action)
            if truck.load < self.min_quantity:
                # if truck is empty we disable all empty nodes
                for t_action in self.action_space.target_space.get_available_actions():
                    if not self.action_space.target_space.is_wait_action(t_action):
                        t_index = self.action_space.target_space.actions_mapping[t_action]
                        t_node = zone_nodes[t_index]
                        if t_node.bikes < self.min_quantity:
                            self.action_space.target_space.disable_action(t_action)

    def prepare_quantity_sub_action(
            self,
            target_node_action: int,
            truck: Truck,
            state_wrapper: StateWrapper,
            current_zone_id: Optional[str],
    ):
        """
        If target_node_action is the wait action, we simply disable all the action quantity actions expect the wait.
        Otherwise, we disable quantities that cannot be dropped (subject to truck load and target node's empty slots)
         and quantities that cannot be picked (subject to truck load and target node's bikes).
        In addition, we populate the TargetNode feature, if not wait
        """
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
                        if q_value < 0:
                            # drop action
                            if not truck.drop_possible(target_node, q_value):
                                self.action_space.quantity_space.disable_action(q_action)
                        else:
                            # pick action
                            if not truck.pick_possible(target_node, q_value):
                                self.action_space.quantity_space.disable_action(q_action)
            else:
                # is wait action, we can only wait
                self.action_space.quantity_space.disable_all_except_wait()


class AgentAbstract:

    def __init__(
            self,
            action_space: Union[RepositionActionSpace, ZoneActionSpace],
            random_state: np.random.RandomState,
            name: AgentType,
            action_spaces: Dict[ActionType, int],
            state_spaces: Dict[StateType, int],
            nodes: List[Node],
            zones: Dict[str, Zone],
            nodes_max_distance: float,
            log: Logger,
            state_builder: StateBuilder,
            is_zone_agent: bool,
            config: Optional[SingleRunConfig] = None,
            mode: RunMode = RunMode.Train,
            **kwargs
    ):
        # internal props
        self.config: SingleRunConfig = config if config is not None else single_run_config
        self.logger: Logger = log
        self.name: AgentType = name
        self.mode: RunMode = mode
        self.is_zone_agent: bool = is_zone_agent
        self.random: np.random.RandomState = random_state
        self.nodes: List[Node] = nodes
        self.zones: Dict[str, Zone] = zones
        self.is_single_zone: bool = True if len(self.zones) == 1 else False
        self.nodes_max_distance: float = nodes_max_distance
        self.action_spaces: Dict[ActionType, int] = action_spaces
        self.state_spaces: Dict[StateType, int] = state_spaces
        self.state_builder: StateBuilder = state_builder
        self.action_space: Union[RepositionActionSpace, ZoneActionSpace] = action_space
        self.replay_buffer: Optional[RolloutBuffer] = None
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stats_tracker: Optional[Tracker] = None
        self.load_generator: Optional[AbstractModel] = None
        self.sub_actions_utils: SubActionUtils = SubActionUtils(
            action_space=self.action_space,
            config=self.config,
            nodes=self.nodes,
            zones=zones,
            nodes_max_distance=nodes_max_distance,
            state_builder=self.state_builder
        )
        # external props
        self.is_baseline: bool = False
        self.requires_evaluation: bool = False
        self.save_agent_state: bool = False
        self.rollout_size: int = 0
        self.continue_training: bool = True
        # running props
        self.choose_action_calls = 0
        self.learn_calls: int = 0
        self.bootstrap_steps: int = 0

    def __str__(self):
        return f'<Agent name={self.name} type={"zone_agent" if self.is_zone_agent else "reposition_agent"} >'

    def set_stats_tracker(self, tracker: Tracker):
        self.stats_tracker: Tracker = tracker
        self.init_extra_tracked_variables()

    def get_action_epsilon(self) -> float:
        return 0

    @property
    def is_bootstrapping(self) -> bool:
        return False

    def is_buffer_full(self, offset: int = 0) -> bool:
        return False

    def set_nodes(self, nodes: List[Node]):
        self.nodes = nodes
        if not self.is_zone_agent:
            self.sub_actions_utils.nodes = nodes

    def _get_action_epsilon(self, t: Step) -> float:
        return 0

    def choose(
            self,
            state_wrapper: StateWrapper,
            t: Step,
            truck: Truck,
            current_zone_id: Optional[str] = None
    ) -> Tuple[Union[RepositionAction, ZoneAction], Optional[Dict[str, Any]]]:

        # for epsilon greedy only
        epsilon = self.get_action_epsilon()
        if self.mode != RunMode.Train:
            epsilon = 0
        random = self.random.random()

        if self.is_zone_agent:
            action = self._choose_zone(
                state_wrapper=state_wrapper,
                t=t,
                truck=truck,
                epsilon=epsilon,
                random=random
            )
        else:
            action = self._choose_action(
                state_wrapper=state_wrapper,
                t=t,
                truck=truck,
                current_zone_id=current_zone_id,
                epsilon=epsilon,
                random=random
            )
        # update action selection stats
        # return the action and the last state update
        action_info: Dict[str, Any] = {
            'wait_movement': False,
            'epsilon': epsilon,
            'random_action': random < epsilon,
            'bootstrapped_action': self.is_bootstrapping,
        }
        self.step_action_info(action_info)
        if not self.is_zone_agent:
            if not self.action_space.target_space.is_wait_action(action.target) \
                    and self.action_space.is_wait_action(action):
                action_info['wait_movement'] = True
                action.target = self.action_space.target_space.wait_action_index
                action.quantity = self.action_space.quantity_space.wait_action_index

        # update running stats
        self._update_running_stats()

        self.choose_action_calls += 1
        return action, action_info

    def step_action_info(self, action_info: Dict[str, Any]):
        pass

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
        # prepare target_node sub-action
        self.sub_actions_utils.prepare_target_node_sub_action(truck, current_zone_id)
        # choose the target node
        t_node_a = self.choose_target_node(
            state_wrapper.get_state(StateType.Target), t, epsilon, random, truck, current_zone_id)
        # prepare the quantity sub-action
        self.sub_actions_utils.prepare_quantity_sub_action(t_node_a, truck, state_wrapper, current_zone_id)
        # chose quantity
        q_a = self.choose_quantity(
            state_wrapper.get_state(StateType.Target), t, epsilon, random, truck, current_zone_id)

        return RepositionAction(target=t_node_a, quantity=q_a)

    def _choose_zone(
            self,
            state_wrapper: StateWrapper,
            t: Step,
            truck: Truck,
            epsilon: float,
            random: float,
            **kwargs
    ) -> ZoneAction:
        # prepare zone action space
        self.sub_actions_utils.prepare_zone_action_space(truck)
        zone_a = self.choose_zone_action(state_wrapper.get_state(StateType.Zone), t, epsilon, random, truck)
        return ZoneAction(zone=zone_a)

    @abstractmethod
    def choose_target_node(
            self,
            state: State,
            t: Step,
            epsilon: float,
            random: float,
            truck: Truck,
            current_zone_id: str,
    ) -> int:
        pass

    @abstractmethod
    def choose_quantity(
            self,
            state: State,
            t: Step,
            epsilon: float,
            random: float,
            truck: Truck,
            current_zone_id: str,
    ) -> int:
        pass

    @abstractmethod
    def choose_zone_action(self, state: State, t: Step, epsilon: float, random: float, truck: Truck) -> int:
        pass

    def evaluate_state_value(self, state_wrapper: StateWrapper) -> Optional[Tensor]:
        return None

    def start_rollout(self):
        pass

    def end_rollout(self, last_value: Tensor, last_done: bool):
        pass

    def _update_running_stats(self):
        pass

    @abstractmethod
    def learn(self):
        pass

    @abstractmethod
    def push_experience(
            self,
            state_wrapper: StateWrapper,
            action: Action,
            reward: float,
            next_state_wrapper: StateWrapper,
            done: bool,
            value: Optional[Tensor] = None,
            log_probs: Optional[Tensor] = None,
            action_mask: Optional[np.ndarray] = None,
            zone_index: Optional[str] = None,
            **kwargs,
    ):
        pass

    def get_model(self, net_type: StateType = None, policy_net=True) -> Optional[torch.nn.Module]:
        return None

    def get_agent_state(self) -> Dict[str, Any]:
        return {}

    def load_agent_state(self, agent_state: Dict[str, Any]):
        pass

    def set_mode(self, mode: RunMode):
        self.mode: RunMode = mode

    def serialize_agent_state(self):
        return pickle.dumps(self.get_agent_state())

    def load_serialized_agent_state(self, serialized_agent_state: bytes):
        deserialized = pickle.loads(serialized_agent_state)
        self.load_agent_state(deserialized)

    def get_prioritized_beta_parameter(self) -> Optional[float]:
        return None

    def init_extra_tracked_variables(self):
        pass

    def get_tracked_learning_params(self) -> Dict[str, str]:
        pass
