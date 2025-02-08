from abc import abstractmethod
from logging import Logger
from typing import Optional, Dict, List

import numpy as np
from torch import Tensor

from cabra import SingleRunConfig
from cabra.core.state import State
from cabra.core.step import Step
from cabra.environment.action_space import RepositionActionSpace, RepositionAction
from cabra.environment.agent import AgentType
from cabra.environment.agent.abstract import AgentAbstract
from cabra.environment.agent.policy import random_policy
from cabra.environment.data_structure import StateType, ActionType
from cabra.environment.node import Node
from cabra.environment.state_builder import StateBuilder
from cabra.environment.state_wrapper import StateWrapper
from cabra.environment.truck import Truck
from cabra.environment.zone import Zone


class BaselineAgent(AgentAbstract):

    def __init__(self,
                 action_space: RepositionActionSpace,
                 random_state: np.random.RandomState,
                 action_spaces: Dict[ActionType, int],
                 state_spaces: Dict[StateType, int],
                 nodes: List[Node],
                 zones: Dict[str, Zone],
                 nodes_max_distance: float,
                 log: Logger,
                 state_builder: StateBuilder,
                 is_zone_agent: bool,
                 name: AgentType,
                 config: Optional[SingleRunConfig] = None,
                 **kwargs):
        super(BaselineAgent, self).__init__(action_space=action_space,
                                            random_state=random_state,
                                            name=name,
                                            action_spaces=action_spaces,
                                            state_spaces=state_spaces,
                                            nodes=nodes,
                                            zones=zones,
                                            nodes_max_distance=nodes_max_distance,
                                            log=log,
                                            is_zone_agent=is_zone_agent,
                                            state_builder=state_builder,
                                            config=config,
                                            **kwargs)
        self.is_baseline = True

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

    def learn(self):
        return None

    def push_experience(
            self,
            state_wrapper: StateWrapper,
            action: RepositionAction,
            reward: float,
            next_state_wrapper: StateWrapper,
            done: bool,
            value: Optional[Tensor] = None,
            log_probs: Optional[Tensor] = None,
            action_mask: Optional[np.ndarray] = None,
    ):
        pass


class RandomAgent(BaselineAgent):
    """
    Random Agent. It randomly selects actions among the available
    """

    def __init__(self,
                 action_space: RepositionActionSpace,
                 random_state: np.random.RandomState,
                 action_spaces: Dict[ActionType, int],
                 state_spaces: Dict[StateType, int],
                 nodes: List[Node],
                 zones: Dict[str, Zone],
                 nodes_max_distance: float,
                 log: Logger,
                 is_zone_agent: bool,
                 state_builder: StateBuilder,
                 config: Optional[SingleRunConfig] = None,
                 **kwargs):
        super(RandomAgent, self).__init__(action_space=action_space,
                                          random_state=random_state,
                                          name=AgentType.Random,
                                          action_spaces=action_spaces,
                                          state_spaces=state_spaces,
                                          nodes=nodes,
                                          zones=zones,
                                          nodes_max_distance=nodes_max_distance,
                                          log=log,
                                          is_zone_agent=is_zone_agent,
                                          state_builder=state_builder,
                                          config=config,
                                          **kwargs)

    def choose_target_node(
            self,
            state: State,
            t: Step,
            epsilon: float,
            random: float,
            truck: Truck,
            current_zone_id: str,
    ) -> int:
        return random_policy(action_space=self.action_space.target_space, random_state=self.random)

    def choose_quantity(
            self,
            state: State,
            t: Step,
            epsilon: float,
            random: float,
            truck: Truck,
            current_zone_id: str,
    ) -> int:
        return random_policy(action_space=self.action_space.quantity_space, random_state=self.random)

    def choose_zone_action(self, state: State, t: Step, epsilon: float, random: float, truck: Truck) -> int:
        return random_policy(action_space=self.action_space.zone_space, random_state=self.random)


class DoNothingAgent(BaselineAgent):
    """
        DoNothing Agent. It always waits. It hopes the system self regulate
        """

    def __init__(self,
                 action_space: RepositionActionSpace,
                 random_state: np.random.RandomState,
                 action_spaces: Dict[ActionType, int],
                 state_spaces: Dict[StateType, int],
                 nodes: List[Node],
                 zones: Dict[str, Zone],
                 nodes_max_distance: float,
                 log: Logger,
                 is_zone_agent: bool,
                 state_builder: StateBuilder,
                 config: Optional[SingleRunConfig] = None,
                 **kwargs):
        super(DoNothingAgent, self).__init__(action_space=action_space,
                                             random_state=random_state,
                                             name=AgentType.DoNothing,
                                             action_spaces=action_spaces,
                                             state_spaces=state_spaces,
                                             nodes=nodes,
                                             zones=zones,
                                             nodes_max_distance=nodes_max_distance,
                                             log=log,
                                             is_zone_agent=is_zone_agent,
                                             state_builder=state_builder,
                                             config=config,
                                             **kwargs)

    def choose_target_node(
            self,
            state: State,
            t: Step,
            epsilon: float,
            random: float,
            truck: Truck,
            current_zone_id: str,
    ) -> int:
        return self.action_space.target_space.wait_action_index

    def choose_quantity(
            self,
            state: State,
            t: Step,
            epsilon: float,
            random: float,
            truck: Truck,
            current_zone_id: str,
    ) -> int:
        return self.action_space.quantity_space.wait_action_index

    def choose_zone_action(self, state: State, t: Step, epsilon: float, random: float, truck: Truck) -> int:
        return self.action_space.zone_space.wait_action_index
