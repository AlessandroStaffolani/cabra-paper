from dataclasses import dataclass
from logging import Logger
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from numpy.random import RandomState
from torch import Tensor

from cabra.common.data_structure import RunMode, BaseEntry, Done
from cabra.common.math_util import normalize_scalar
from cabra.common.stats_tracker import Tracker
from cabra.core.step import Step
from cabra.core.step_data import StepData
from cabra.core.timestep import TimeStep, time_step_factory_get, time_step_factory_reset, SimulationEnded
from cabra.emulator.models import create_model_from_type
from cabra.emulator.models.abstract import AbstractModel
from cabra.environment import logger, EnvConfig
from cabra.environment.action_space import SubActionSpace, RepositionAction, RepositionActionSpace, \
    ContinuousRepositionActionSpace, PPORepositionActionSpace, ZoneAction
from cabra.environment.agent import AgentType
from cabra.environment.agent.experience_replay.experience_entry import RawTransition
from cabra.environment.data_structure import StateType, StateFeatureName, ActionType
from cabra.environment.node import Node, init_nodes, DistancesProvider
from cabra.environment.reward import RewardAbstract, get_reward_class
from cabra.environment.state_builder import StateBuilder
from cabra.environment.state_wrapper import StateWrapper
from cabra.environment.truck import Truck, init_trucks
from cabra.run.config import SingleRunConfig


def get_action_space_class(agent_type: AgentType, use_continuous: bool):
    if agent_type.is_ppo() and not use_continuous:
        return PPORepositionActionSpace
    elif agent_type.is_ppo() and use_continuous:
        return ContinuousRepositionActionSpace
    else:
        return RepositionActionSpace


class EnvNotReadyError(Exception):

    def __init__(self, method, *args):
        super(EnvNotReadyError, self).__init__(
            f'Trying to call env "{method}" method on a not ready environment. You must call the reset method before',
            *args
        )


class ActionInvalidError(Exception):

    def __init__(self, action: RepositionAction, field: str = None, *args):
        message = f'Action is invalid.'
        if field is not None:
            message += f' {field} value is incorrect.'
        message += f' {action}'
        super(ActionInvalidError, self).__init__(message, *args)


class ActionDisabledError(Exception):

    def __init__(self, action: RepositionAction, sub_action: str, value: int, *args):
        message = f'Action {value} for sub action {sub_action} is currently disabled. RepositionAction: {action}'
        super(ActionDisabledError, self).__init__(message, *args)


class ActionMoveSameNodeError(Exception):

    def __init__(self, action: RepositionAction, *args):
        message = f'Trying to move quantity from the same node. RepositionAction = {action}'
        super(ActionMoveSameNodeError, self).__init__(message, *args)


class ActionMovementInvalid(Exception):

    def __init__(self, node: Node, quantity: int, move_type: str, *args):
        if move_type == 'allocate':
            message = f'Node {node.name} cannot receive {quantity} bikes, it has not enough empty spaces'
        elif move_type == 'remove':
            message = f'Node {node.name} cannot give {quantity} bikes, it has not enough bikes available'
        else:
            message = f'Invalid movement for node {node.name}'
        super(ActionMovementInvalid, self).__init__(message, *args)


def _validate_single_sub_action(field, value: int, action: RepositionAction, action_space: SubActionSpace):
    try:
        if not action_space.is_action_available(value):
            raise ActionDisabledError(action, sub_action=field, value=value)
    except IndexError:
        raise ActionInvalidError(action, field=field)


@dataclass
class PartialTransition(BaseEntry):
    zone_state_wrapper: StateWrapper
    state_wrapper: StateWrapper
    zone_action: ZoneAction
    action: RepositionAction
    truck: Truck
    action_cost: float
    action_shortages: int
    penalty: float
    zone_value: Tensor
    zone_log_probs: Tensor
    zone_action_mask: Tuple[np.ndarray, ...]
    value: Tensor
    log_probs: Tensor
    action_mask: Tuple[np.ndarray, ...]
    policy_index: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'zone_state_wrapper': self.zone_state_wrapper,
            'state_wrapper': self.state_wrapper,
            'zone_action': self.zone_action,
            'action': self.action,
            'truck': self.truck,
            'action_cost': self.action_cost,
            'action_shortages': self.action_shortages,
            'penalty': self.penalty,
            'zone_value': self.zone_value,
            'zone_log_probs': self.zone_log_probs,
            'zone_action_mask': self.zone_action_mask,
            'value': self.value,
            'log_probs': self.log_probs,
            'action_mask': self.action_mask,
            'policy_index': self.policy_index,
        }


class EnvWrapper:

    def __init__(
            self,
            run_code: str,
            config: SingleRunConfig,
            random_state: RandomState,
            log: Optional[Logger] = None,
            test_mode: bool = False,
            run_mode: RunMode = RunMode.Train,
            random_seed: int = 42,
    ):
        self.config: SingleRunConfig = config
        self.env_config: EnvConfig = config.environment
        self.logger: Logger = log if log is not None else logger
        self.random: RandomState = random_state
        self.random_seed: int = random_seed
        self.run_code: str = run_code
        # public properties
        self.use_virtual_reset: bool = self.env_config.use_virtual_reset
        self.run_mode: RunMode = run_mode
        self.nodes: List[Node] = []
        self.nodes_name_index: Dict[str, int] = {}
        self.nodes_max_distance: float = 0
        self.distances_provider: DistancesProvider = DistancesProvider(
            config=self.config)
        self.trucks: List[Truck] = []
        self.time_step: Optional[TimeStep] = None
        self.load_generator: Optional[AbstractModel] = None
        self.stats_tracker: Optional[Tracker] = None
        self.state_builder: Optional[StateBuilder] = None
        self.action_spaces: Dict[ActionType, int] = {
            ActionType.Target: 0,
            ActionType.Quantity: 0,
        }
        self.state_spaces: Dict[StateType, int] = {
            StateType.Target: 0,
            StateType.Quantity: 0,
        }
        self.state_types: List[StateType] = []
        self.action_space: Optional[RepositionActionSpace] = None
        self.reward_class: Optional[RewardAbstract] = None
        self.initialized = False
        self.ready = False
        self.test_mode = test_mode
        self.use_continuous: bool = self.env_config.action_space.use_continuous_action_space
        # running props
        self.current_state_wrapper: Optional[StateWrapper] = None
        self.current_demand: Optional[StepData] = None
        self.shortages_since_last_reward: int = 0
        self.env_shortages_since_last_reward: int = 0
        # init internal components
        self.init()

    def __str__(self):
        return f'<EnvWrapper nodes={len(self.nodes)} time_steps={self.time_step.stop_step} seed={self.random_seed} >'

    def init(self):
        """
        Init all the internal components of the environment
        """
        self._init_nodes()
        self._init_trucks()
        self._init_time_step()
        self._init_load_generator()
        self._init_action_space()
        self._init_reward_class()
        self._init_state_builder()
        self._init_spaces()
        self.initialized = True

    def _init_nodes(self):
        n_nodes = self.config.environment.nodes.n_nodes
        self.nodes = init_nodes(nodes_config=self.config.environment.nodes, random_state=self.random,
                                predicted_demand=None, use_prediction=self.env_config.nodes.use_predictions)
        self.nodes_max_distance = self.distances_provider.get_max_distance(self.nodes)
        self.distances_provider.set_node_name_index_mapping(self.nodes)
        self.nodes_name_index = {n.name: n.index for n in self.nodes}

        if self.env_config.nodes.loaded_config.normalize_coordinates:
            min_lat, min_lng = self.distances_provider.min_position.lat, self.distances_provider.min_position.lng
            max_lat, max_lng = self.distances_provider.max_position.lat, self.distances_provider.max_position.lng
            for node in self.nodes:
                node.position.lat = normalize_scalar(node.position.lat, min_val=min_lat, max_val=max_lat)
                node.position.lng = normalize_scalar(node.position.lng, min_val=min_lng, max_val=max_lng)

        assert len(self.nodes) == n_nodes

    def _init_trucks(self):
        self.trucks = init_trucks(
            config=self.config,
            random_state=self.random,
            initial_node=self.nodes[0],
            distances_provider=self.distances_provider
        )
        assert len(self.trucks) == self.config.environment.trucks.n_trucks

    def _init_time_step(self):
        config = self.config.environment.time_step
        self.time_step: TimeStep = time_step_factory_get(
            run_code=self.run_code,
            step_per_second=config.step_per_second,
            step_size=config.step_size,
            stop_step=config.stop_step,
            stop_date=config.stop_date,
            initial_date=config.initial_date,
            logger=self.logger
        )

        # if AgentType.is_policy_gradient(self.config.environment.agent.type) or self.config.run.use_on_policy_agent:
        #     self.time_step.stop_step += self.time_step.step_size
        # # multiply the time steps for the number of state stacked
        # if self.config.environment.state.stack_n_states > 1:
        #     self.time_step.stop_step += self.time_step.step_size
        # self.time_step.stop_step *= self.config.environment.state.stack_n_states

    def _init_load_generator(self):
        self.load_generator: AbstractModel = create_model_from_type(
            model_type=self.config.emulator.model.type,
            nodes=self.nodes,
            em_config=self.config.emulator,
            random_state=self.random,
            log=self.logger,
            run_mode=self.run_mode,
            random_seed=self.random_seed
        )

    def _init_action_space(self):
        action_space_class = get_action_space_class(
            agent_type=self.env_config.agent.type,
            use_continuous=self.env_config.action_space.use_continuous_action_space
        )
        self.action_space: RepositionActionSpace = action_space_class(
            n_nodes=self.env_config.nodes.n_nodes,
            n_quantities=self.env_config.trucks.capacity,
            min_quantity=self.env_config.action_space.min_quantity,
            add_wait_space=self.env_config.action_space.add_wait_space
        )

    def _init_reward_class(self):
        self.reward_class: RewardAbstract = get_reward_class(
            reward_type=self.env_config.reward.type,
            config=self.config,
            n_nodes=self.env_config.nodes.n_nodes,
            disable_cost=self.env_config.reward.disable_cost,
            run_mode=self.run_mode,
            nodes=self.nodes,
            distances_provider=self.distances_provider,
            **self.env_config.reward.parameters,
        )

    def _init_state_builder(self):
        self.state_builder: StateBuilder = StateBuilder(
            config=self.env_config,
            nodes=self.nodes,
            trucks=self.trucks,
            distances_provider=self.distances_provider,
            action_space=self.action_space
        )
        self.state_types = [StateType.Target, StateType.Quantity]

    def _init_spaces(self):
        # action space
        self.action_spaces = self.action_space.get_sizes()
        # states space
        self.state_spaces[StateType.Target] = self.state_builder.get_state_size(StateType.Target)
        self.state_spaces[StateType.Quantity] = self.state_builder.get_state_size(StateType.Quantity)

    def set_stats_tracker(self, tracker: Tracker):
        self.stats_tracker: Tracker = tracker

    @property
    def current_time_step(self) -> Step:
        return self.time_step.current_step

    @property
    def is_last_time_step(self) -> bool:
        return self.time_step.is_last

    def info(self):
        """
        Returns
        -------
        the information of the environment
        """
        return {
            'n_nodes': len(self.nodes),
            'n_trucks': len(self.trucks),
            'state_features': self.state_builder.get_all_features(),
            'last_step': self.time_step.current_step,
            'load_generator': self.load_generator.model_name(),
            'initialized': self.initialized,
        }

    def apply_action(
            self,
            action: RepositionAction,
            truck: Truck
    ) -> Tuple[float, int, float]:
        action_cost, penalty, caused_shortage = self.do_reposition(action, truck)
        if penalty == 0:
            self.apply_action_on_state(action, truck)
        # shortages generated by the action
        shortages = 1 if caused_shortage else 0
        return action_cost, shortages, penalty

    def step(
            self,
            partial_transitions: List[PartialTransition],
    ) -> Tuple[List[RawTransition], Done]:
        completed_transitions: List[RawTransition] = []
        self.next_step(increase_step=True)
        env_shortages = self.update_current_demand()
        self.env_shortages_since_last_reward += env_shortages
        # we count also the shortages we have after demand changes
        self.shortages_since_last_reward += self._n_shortages()
        # compute next state
        next_state_wrapper = self.get_state()
        # check if done
        done = self._episode_is_done()
        # complete transitions
        for s_wrapper, action, truck, cost, n_shortages, penalty, value, log_probs, action_mask in partial_transitions:
            # TODO Verify this
            # truck shortages are the shortages caused by truck action,
            # plus the shortages caused by empty steps, plus the shortages we observe after demand changes
            # perhaps, shortages are only the shortages we observe after each round of actions,
            # so it is only the shortages caused by empty steps, plus the shortages we observe after demand changes
            truck_shortages = self.shortages_since_last_reward + n_shortages
            self.update_truck_features(truck, next_state_wrapper)
            # compute reward
            reward = self.reward_class.compute(
                state_wrapper=s_wrapper,
                action_cost=cost,
                step=self.time_step.current_step,
                nodes=self.nodes,
                shortages=truck_shortages,
                env_shortages=self.env_shortages_since_last_reward,
                penalty=penalty,
            )
            completed_transitions.append(RawTransition(
                state_wrapper=s_wrapper,
                action=action,
                reward=reward,
                next_state_wrapper=next_state_wrapper.copy(),
                done=done.to_bool(),
                value=value,
                log_probs=log_probs,
                action_mask=action_mask,
                step_info=self.prepare_step_info(truck_shortages)
            ))

        self.shortages_since_last_reward = 0
        self.env_shortages_since_last_reward = 0

        return completed_transitions, done

    def empty_step(self) -> Done:
        """
        Called when all the trucks are busy, used only to update the demand and compute the number of shortages
        """
        # get env_shortages from new current demand
        self.next_step(increase_step=True)
        env_shortages = self.update_current_demand()
        shortages = self._n_shortages()
        self.shortages_since_last_reward += shortages
        self.env_shortages_since_last_reward += env_shortages
        self.unmask_actions()
        done = self._episode_is_done()
        return done

    def get_state(self) -> StateWrapper:
        state_wrapper = StateWrapper(
            state_types=self.state_types,
            state_builder=self.state_builder,
            current_step=self.current_time_step,
            current_demand=self.current_demand,
        )

        self.current_state_wrapper = state_wrapper
        return state_wrapper

    def do_reposition(self, action: RepositionAction, truck: Truck, *args, **kwargs) -> Tuple[float, float, bool]:
        wait, target_node_index, quantity_val = self.action_space.action_to_action_value(action)
        penalty = 0
        action_cost = 0
        caused_shortage = False
        if not wait:
            target_node = self.nodes[target_node_index]
            was_in_shortage = self.nodes[target_node_index].is_in_shortage
            # we need to move some bikes
            if self.is_action_possible(wait, target_node, quantity_val, truck):
                truck.reposition(target_node=target_node, quantity=quantity_val, step=self.current_time_step)
                action_cost = self.compute_action_cost(action, truck)
                penalty = 0
                now_is_shortage = target_node.is_in_shortage
            else:
                penalty = self.env_config.reward.invalid_action_penalty
                now_is_shortage = False
            if not was_in_shortage and now_is_shortage:
                caused_shortage = True

        return action_cost, penalty, caused_shortage

    def apply_action_on_state(self, action: RepositionAction, truck: Truck) -> StateWrapper:
        for _, state in self.current_state_wrapper.items():
            next_step = self.current_time_step.add(Step.from_total_steps(self.time_step.step_size))
            self.state_builder.apply_action_on_state(state, next_step, action, truck, self.action_space)
        self.current_state_wrapper.set_current_truck(truck)
        return self.current_state_wrapper

    def update_current_demand(self) -> int:
        if self.env_config.nodes.use_predictions:
            next_episode_step = self.current_time_step.add(Step.from_total_steps(self.time_step.step_size))
        else:
            next_episode_step = None
        self.current_demand, env_shortages = self.load_generator.update_data(
            current_step=self.current_time_step,
            next_step=next_episode_step
        )
        total_bikes = sum([n.bikes for n in self.nodes])
        total_bikes += sum([n.empty_slots for n in self.nodes])
        total_bikes_true = sum([n.total_slots for n in self.nodes])
        assert total_bikes == total_bikes_true
        return env_shortages

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

    def disable_truck_unavailable_actions(self, truck: Truck):
        for q in range(truck.empty_slots, truck.capacity):
            # cannot pick more than empty slots, so we disable all quantities bigger than emmpty_slots
            q_val = q + 1
            if q_val > self.config.environment.action_space.min_quantity:
                action_index = self.action_space.quantity_space.inverted_actions_mapping[q_val]
                self.action_space.quantity_space.disable_action(action_index)
        for q in range(truck.load, truck.capacity):
            # cannot drop more than load quantity, so we disable quantities bigger than load
            q_val = -(q + 1)
            if abs(q_val) > self.config.environment.action_space.min_quantity:
                action_index = self.action_space.quantity_space.inverted_actions_mapping[q_val]
                self.action_space.quantity_space.disable_action(action_index)

    def update_truck_features(self, current_truck: Truck, state_wrapper: StateWrapper) -> StateWrapper:
        new_trucks_features = self.state_builder.build_state_feature_values(
            feature=StateFeatureName.TruckFeatures,
            current_step=self.current_time_step,
            current_demand=self.current_demand
        )
        state_wrapper.update_feature_value(
            feature_name=new_trucks_features[0].name, new_value=new_trucks_features[0].value)
        state_wrapper.set_current_truck(current_truck)
        # if previous state and action are used set them in the current state
        if StateFeatureName.PreviousState in self.state_builder.common_features:
            state_wrapper.set_previous_state(self.state_builder, current_truck)
        if StateFeatureName.PreviousAction in self.state_builder.common_features:
            state_wrapper.set_previous_action(self.state_builder, current_truck)
        return state_wrapper

    def store_previous_feature_values(self, action: RepositionAction, state_wrapper: StateWrapper, truck: Truck):
        for state_type, state in state_wrapper.items():
            features = state.features_names()
            need_slice = False
            if StateFeatureName.PreviousState.value in features:
                features.remove(StateFeatureName.PreviousState.value)
                need_slice = True
            if StateFeatureName.PreviousAction in features:
                features.remove(StateFeatureName.PreviousAction)
                need_slice = True
            if need_slice:
                previous_state = state.get_state_slice(state, features)
            else:
                previous_state = state.copy()
            self.state_builder.set_previous_feature(state_type, truck, StateFeatureName.PreviousState, previous_state)
            self.state_builder.set_previous_feature(state_type, truck, StateFeatureName.PreviousAction, action)

    def prepare_step_info(self, shortages: int) -> Dict[str, Any]:
        solved_step = False
        if shortages == 0 and self.env_shortages_since_last_reward == 0:
            solved_step = True
        step_info = {
            'solved_step': solved_step,
            'reward_info': self.reward_class.reward_info()
        }
        return step_info

    def next_step(self, increase_step: bool = True) -> Step:
        if increase_step:
            if self.time_step.is_last:
                raise SimulationEnded(step=self.time_step.current_step)
            else:
                return self.time_step.next()
        else:
            return self.time_step.current_step

    def _episode_is_done(self) -> Done:
        generator_done = self.load_generator.is_done(self.current_time_step)
        return generator_done

    def _n_shortages(self) -> int:
        count = 0
        for node in self.nodes:
            if node.is_in_shortage:
                count += 1
        return count

    def is_action_possible(
            self,
            wait: bool,
            target_node: Node,
            quantity_val: int,
            truck: Truck,
            *args,
            **kwargs) -> bool:
        possible = True
        if not wait:
            if quantity_val < 0:
                possible = truck.drop_possible(node=target_node, quantity=quantity_val)
            else:
                possible = truck.pick_possible(node=target_node, quantity=quantity_val)
        return possible

    def validate_action(self, action: RepositionAction):
        target, quantity = action
        _validate_single_sub_action(ActionType.Target.value, target, action, self.action_space.target_space)
        _validate_single_sub_action(ActionType.Quantity.value, quantity, action, self.action_space.quantity_space)

    def unmask_actions(self):
        self.action_space.unmask_all()

    def reset(
            self,
            done: Optional[Done] = None,
            reset_time_step=True,
            reset_generator=True,
            reset_reward=True,
            reset_nodes=True,
            reset_trucks=True,
            show_log=True
    ) -> StateWrapper:
        if done is not None:
            assert done.to_bool() is True
        if self.use_virtual_reset:
            if done == Done.VirtualDone:
                reset_nodes = False
                reset_trucks = False
                reset_time_step = False
        self._reset_env(reset_time_step, reset_generator, reset_reward, reset_nodes, reset_trucks, show_log)
        if self.env_config.nodes.use_predictions:
            next_episode_step = self.current_time_step.add(Step.from_total_steps(self.time_step.step_size))
        else:
            next_episode_step = None
        self.load_generator.update_prediction_only(next_episode_step)
        self.current_demand, _ = self.load_generator.update_data(
            current_step=self.current_time_step,
            next_step=self.current_time_step
        )
        state: StateWrapper = self.get_state()
        self.ready = True
        return state

    def _reset_env(
            self,
            reset_time_step: bool,
            reset_generator: bool,
            reset_reward: bool,
            reset_nodes: bool,
            reset_trucks: bool,
            show_log: bool
    ):
        if reset_time_step:
            self._reset_time_step(show_log)
        if reset_generator:
            self.load_generator.reset(reset_nodes=reset_nodes)
        if reset_reward:
            self.reward_class.reset()
        if reset_trucks:
            for truck in self.trucks:
                truck.reset()
        # reset running props
            self.current_state_wrapper: Optional[StateWrapper] = None
        self.shortages_since_last_reward: int = 0
        self.env_shortages_since_last_reward: int = 0
        self.current_demand: Optional[StepData] = None

    def _reset_time_step(self, show_log=True):
        config = self.env_config.time_step
        self.time_step: TimeStep = time_step_factory_reset(
            run_code=self.run_code,
            recreate=True,
            step_per_second=config.step_per_second,
            step_size=config.step_size,
            stop_step=config.stop_step,
            stop_date=config.stop_date,
            initial_date=config.initial_date,
            logger=self.logger,
            show_log=show_log
        )
        # if AgentType.is_policy_gradient(self.env_config.agent.type) or self.config.run.use_on_policy_agent:
        #     self.time_step.stop_step += self.time_step.step_size
        # if self.env_config.state.stack_n_states > 1:
        #     self.time_step.stop_step += self.time_step.step_size
        # self.time_step.stop_step *= self.env_config.state.stack_n_states
