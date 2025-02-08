import itertools
import time

from abc import abstractmethod
from dataclasses import dataclass
from logging import Logger
from typing import Optional, Dict, List

import numpy as np
from torch import Tensor

from cabra import SingleRunConfig
from cabra.common.data_structure import RunMode
from cabra.common.math_util import normalize_scalar
from cabra.core.state import State
from cabra.core.step import Step
from cabra.environment.action_space import RepositionActionSpace, RepositionAction
from cabra.environment.agent import AgentType
from cabra.environment.agent.abstract import AgentAbstract
from cabra.environment.agent.baseline import BaselineAgent
from cabra.environment.agent.baseline.constrained import ConstrainedSubActionUtils
from cabra.environment.agent.policy import random_policy
from cabra.environment.data_structure import StateType, ActionType
from cabra.environment.node import Node, DistancesProvider
from cabra.environment.reward import get_reward_class, GlobalShortageAndCostReward
from cabra.environment.state_builder import StateBuilder
from cabra.environment.state_wrapper import StateWrapper
from cabra.environment.truck import Truck
from cabra.environment.zone import Zone


class SolutionMissingException(Exception):

    def __init__(self, *args):
        super().__init__('Solution object is not defined', *args)


@dataclass
class Solution:
    target_node: Optional[int] = None
    quantity: Optional[int] = None

    def get_sub_action(self, action_type: ActionType) -> int:
        if action_type == ActionType.Target:
            return self.target_node
        elif action_type == ActionType.Quantity:
            return self.quantity
        else:
            return -1

    def to_action(self) -> RepositionAction:
        return RepositionAction(target=self.target_node, quantity=self.quantity)


class SAPowerfulAgent(BaselineAgent):
    """
    Simulated Annealing Agent.
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
        super(SAPowerfulAgent, self).__init__(action_space=action_space,
                                              random_state=random_state,
                                              name=AgentType.SimulatedAnnealing,
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
        # parameters
        self.max_iterations: int = self.config.environment.agent.simulated_annealing.max_iterations
        self.cooling_rate: float = self.config.environment.agent.simulated_annealing.cooling_rate
        # internal objects
        self.shortage_weight: float = self.config.environment.reward.parameters['shortage_weight']
        self.environment_shortage_weight: float = self.config.environment.reward.parameters[
            'environment_shortage_weight']
        self.cost_weight: float = self.config.environment.reward.parameters['cost_weight']

        self.current_solution: Optional[Solution] = None

    def choose_target_node(self, state: State, t: Step, epsilon: float, random: float, truck: Truck,
                           current_zone_id: str) -> int:
        self.prepare_solution(t, truck)
        return self.return_solution(ActionType.Target)

    def choose_quantity(self, state: State, t: Step, epsilon: float, random: float, truck: Truck,
                        current_zone_id: str) -> int:
        return self.return_solution(ActionType.Quantity)

    def prepare_solution(self, t: Step, action_truck: Truck):
        # copy truck and nodes in order to allow safe updates
        start_time = time.time()
        truck = action_truck.copy()
        nodes = [node.copy() for node in self.nodes]

        t_initial = self.max_iterations  # Initial temperature
        t_final = 1  # Final temperature
        t_current = t_initial
        # Get the list of all available actions between target node and quantity
        possible_solutions = self.prepare_possible_actions_structure()

        # sample one action and remove it from the set of possible solutions
        current_solution = self.sample_solution(possible_solutions, truck, nodes)
        # compute the score, aka the reward of the given solution
        current_score = self.evaluate_solution(t, current_solution, truck, nodes)
        best_solution = current_solution
        best_score = current_score

        while t_current > t_final and len(possible_solutions) > 0:
            # get a new random solution
            neighbor_solution = self.sample_solution(possible_solutions, truck, nodes)
            # compute the score of the new solution
            neighbor_score = self.evaluate_solution(t, neighbor_solution, truck, nodes)

            # Calculate the change in score
            delta_score = neighbor_score - current_score

            if delta_score > 0:
                # Neighbor solution is better; accept it
                accept = True
            else:
                # Neighbor solution is worse; accept it with a probability
                acceptance_probability = np.exp(delta_score * 10 / t_current)
                accept = self.random.random() < acceptance_probability

            if accept:
                current_score = neighbor_score
                # Update the best solution found so far
                if neighbor_score > best_score:
                    best_solution = neighbor_solution
                    best_score = neighbor_score

            # Cool down the temperature
            t_current *= self.cooling_rate

        self.current_solution = best_solution
        # track metrics
        solution_time = time.time() - start_time
        unexplored_actions = sum([len(arr.compressed()) for _, arr in possible_solutions.items()])
        self.stats_tracker.track(
            f'{RunMode.Eval}-{self.config.random_seeds.evaluation[0]}/history/single_truck_action_time',
            solution_time
        )
        self.stats_tracker.track(
            f'{RunMode.Eval}-{self.config.random_seeds.evaluation[0]}/history/actions_not_explored',
            unexplored_actions
        )

    def return_solution(self, action_type: ActionType) -> int:
        if self.current_solution is None:
            raise SolutionMissingException
        return self.current_solution.get_sub_action(action_type)

    def choose_zone_action(self, state: State, t: Step, epsilon: float, random: float, truck: Truck) -> int:
        raise NotImplementedError(f'{self.__name__} is not available in multi zone mode.')

    def evaluate_solution(self, t: Step, solution: Solution, truck: Truck, nodes: List[Node]) -> float:
        evaluation_truck = truck.copy()
        action = solution.to_action()
        wait, target_node_index, quantity_val = self.action_space.action_to_action_value(action)
        action_cost = 0
        original_target_node = Node
        if not wait:
            # apply the action on the copied node and compute the action cost
            original_target_node = nodes[target_node_index]
            target_node = original_target_node.copy()
            nodes[target_node_index] = target_node

            evaluation_truck.reposition(target_node=target_node, quantity=quantity_val, step=t)
            action_cost = self.compute_action_cost(action, evaluation_truck)

        # compute shortages
        shortages = self.compute_shortages(nodes)
        # compute the reward of the solution
        reward = self.compute_solution_reward(action_cost, shortages, 0)
        # rollback changes to node
        if not wait and original_target_node is not None:
            nodes[target_node_index] = original_target_node
        return reward

    def compute_action_cost(self, action: RepositionAction, truck: Truck) -> float:
        wait, target_node_index, quantity_val = self.action_space.action_to_action_value(action)
        if wait:
            return 0
        else:
            reposition_time = truck.reposition_time
            max_reposition_time = truck.max_reposition_time
            if self.config.environment.reward.parameters['normalize_cost']:
                action_cost = normalize_scalar(value=reposition_time, max_val=max_reposition_time, min_val=0)
            else:
                action_cost = reposition_time
            return action_cost

    def compute_shortages(self, nodes: list[Node]) -> int:
        count = 0
        for node in nodes:
            if node.is_in_shortage:
                count += 1
        return count

    def compute_solution_reward(self, action_cost: float, shortages: int, env_shortages: int) -> float:
        env_shortages_rew = -(self.environment_shortage_weight * env_shortages)
        shortages_rew = -(shortages * self.shortage_weight)
        cost_rew = -(action_cost * self.cost_weight)
        reward = env_shortages_rew + shortages_rew + cost_rew
        return reward

    def stop_condition(self, iterations, all_actions: list[tuple[int, int]]) -> bool:
        if iterations >= self.max_iterations or len(all_actions) == 0:
            return True
        else:
            return False

    def sample_solution(
            self,
            possible_solutions: dict[int, np.ma.masked_array],
            truck: Truck,
            nodes: List[Node]
    ) -> Solution:
        # sample target node action
        target_node_index = self.random.choice(list(possible_solutions.keys()))

        # disable actions that are not available on this node
        if not self.action_space.target_space.is_wait_action(target_node_index):
            target_node = nodes[target_node_index]
            for q_action in possible_solutions[target_node_index].compressed():
                if not self.action_space.quantity_space.is_wait_action(q_action):
                    q_value = self.action_space.quantity_space.actions_mapping[q_action]
                    if q_value < 0:
                        # drop action
                        if not truck.drop_possible(target_node, q_value):
                            possible_solutions[target_node_index].mask[q_action] = True
                    else:
                        # pick action
                        if not truck.pick_possible(target_node, q_value):
                            possible_solutions[target_node_index].mask[q_action] = True

        quantity = self.random.choice(possible_solutions[target_node_index].compressed())
        # disable the quantity action
        if self.action_space.target_space.is_wait_action(target_node_index):
            possible_solutions[target_node_index].mask[0] = True
        else:
            possible_solutions[target_node_index].mask[quantity] = True
        if len(possible_solutions[target_node_index].compressed()) == 0:
            # the entire target node has not actions anymore, we remove it
            del possible_solutions[target_node_index]

        return Solution(
            target_node=target_node_index,
            quantity=quantity
        )

    def prepare_possible_actions_structure(self) -> dict[int, np.ma.masked_array]:
        possible_actions = {}
        for target_action in self.action_space.target_space.get_available_actions():
            if not self.action_space.target_space.is_wait_action(target_action):
                possible_actions[target_action] = np.ma.array(
                    data=self.action_space.quantity_space.get_all_actions().copy(),
                    mask=self.action_space.quantity_space.get_mask().copy()
                )
            else:
                possible_actions[target_action] = np.ma.array(
                    data=[self.action_space.quantity_space.wait_action_index],
                    mask=[False]
                )
        return possible_actions

    def init_extra_tracked_variables(self):
        metrics = ['single_truck_action_time', 'actions_not_explored']
        for m in metrics:
            for seed in self.config.random_seeds.evaluation:
                self.stats_tracker.init_tracking(f'{RunMode.Eval}-{seed}/history/{m}',
                                                 tensorboard=False, redis_save=False, aggregation_fn=np.mean,
                                                 str_precision=6)


class ConstrainedSAPowerfulAgent(SAPowerfulAgent):

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
                 distances_provider: DistancesProvider,
                 config: Optional[SingleRunConfig] = None,
                 **kwargs):
        super(ConstrainedSAPowerfulAgent, self).__init__(action_space=action_space,
                                                         random_state=random_state,
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
        self.name = AgentType.ConstrainedSimulatedAnnealing
        self.sub_actions_utils: ConstrainedSubActionUtils = ConstrainedSubActionUtils(
            action_space=self.action_space,
            config=self.config,
            nodes=self.nodes,
            zones=zones,
            nodes_max_distance=nodes_max_distance,
            max_distance=nodes_max_distance * 20,
            zone_max_distance=nodes_max_distance * 20,
            critical_threshold=self.nodes[0].critical_threshold,
            zones_filtered_size=self.config.environment.constrained_space.zones_filtered_size,
            state_builder=self.state_builder,
            distances_provider=distances_provider,
        )
