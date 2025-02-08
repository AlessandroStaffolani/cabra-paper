import json
from logging import Logger
from typing import List, Dict, Optional, Tuple

from numpy.random import RandomState

from cabra.common.data_structure import RunMode, Done
from cabra.common.distance_helper import Position
from cabra.core.step import Step
from cabra.environment.action_space import RepositionAction, RepositionActionSpace, ZoneActionSpace, \
    ZoneAction
from cabra.environment.agent.experience_replay.experience_entry import RawTransition
from cabra.environment.data_structure import StateType, StateFeatureName
from cabra.environment.state_builder import StateBuilder
from cabra.environment.state_wrapper import StateWrapper
from cabra.environment.truck import Truck
from cabra.environment.wrapper import EnvWrapper, get_action_space_class, PartialTransition
from cabra.environment.zone import Zone
from cabra.run.config import SingleRunConfig


class ZonesEnvWrapper(EnvWrapper):

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
        # internal props
        self.zone_state_type: StateType = StateType.Zone
        self.zones: Dict[str, Zone] = {}
        self.zone_action_space: Optional[ZoneActionSpace] = None
        super().__init__(run_code, config, random_state, log, test_mode, run_mode, random_seed)
        self.zones_max_size: int = 0
        # running props
        self.current_zone: Optional[Zone] = None
        self.current_zone_state_wrapper: Optional[StateWrapper] = None

    def __str__(self):
        return f'<ZoneEnvWrapper nodes={len(self.nodes)} time_steps={self.time_step.stop_step} seed={self.random_seed} >'

    @property
    def n_zones(self) -> int:
        return len(self.zones)

    def init(self):
        self._init_nodes()
        self._init_zones()
        self._init_trucks()
        self._init_time_step()
        self._init_load_generator()
        self._init_action_space()
        self._init_reward_class()
        self._init_state_builder()
        self._init_spaces()
        self.initialized = True

    def _init_zones(self):
        with open(self.env_config.nodes.loaded_config.get_zones_load_path(), 'r') as f:
            zones_data = json.load(f)
        for node in self.nodes:
            if node.zone not in self.zones:
                centroid = zones_data[node.zone]['centroid']
                self.zones[node.zone] = Zone(node.zone, [node], Position(lat=centroid[1], lng=centroid[0]))
            else:
                self.zones[node.zone].add_node(node)
        self.zones_max_size = max([zone.max_zone_size for _, zone in self.zones.items()])
        for _, zone in self.zones.items():
            zone.max_zone_size = self.zones_max_size
        self.is_single_zone: bool = True if len(self.zones) == 1 else False

    def _init_action_space(self):
        action_space_class = get_action_space_class(
            agent_type=self.env_config.agent.type,
            use_continuous=self.env_config.action_space.use_continuous_action_space
        )
        self.action_space: RepositionActionSpace = action_space_class(
            n_nodes=self.zones_max_size,
            n_quantities=self.env_config.trucks.capacity,
            min_quantity=self.env_config.action_space.min_quantity,
            add_wait_space=self.env_config.action_space.add_wait_space
        )
        self.zone_action_space: ZoneActionSpace = ZoneActionSpace(
            n_zones=self.n_zones,
            zones_mapping={i: zone_id for i, zone_id in enumerate(list(self.zones.keys()))},
            add_wait_action=self.env_config.zones.action_space_can_wait
        )

    def _init_state_builder(self):
        self.state_builder: StateBuilder = StateBuilder(
            config=self.env_config,
            nodes=self.nodes,
            trucks=self.trucks,
            zones=self.zones,
            distances_provider=self.distances_provider,
            action_space=self.action_space,
            zone_action_space=self.zone_action_space
        )
        self.state_types = [StateType.Target]

    def _init_spaces(self):
        # action space
        self.action_spaces = self.action_space.get_sizes()
        zone_spaces = self.zone_action_space.get_sizes()
        for a_type, size in zone_spaces.items():
            self.action_spaces[a_type] = size
        # states space
        self.state_spaces[StateType.Target] = self.state_builder.get_state_size(StateType.Target)
        self.state_spaces[self.zone_state_type] = self.state_builder.get_state_size(self.zone_state_type)

    def apply_reposition_action(
            self,
            action: RepositionAction,
            truck: Truck,
            current_zone_id: str,
    ) -> Tuple[float, int, float]:
        action_cost, penalty, caused_shortage = self.do_reposition(action, truck, current_zone_id)
        # shortages generated by the action
        shortages = 1 if caused_shortage else 0
        return action_cost, shortages, penalty

    def do_reposition(self, action: RepositionAction, truck: Truck, current_zone_id: str) -> Tuple[float, float, bool]:
        wait, target_node_index, quantity_val = self.action_space.action_to_action_value(action)
        penalty = 0
        action_cost = 0
        caused_shortage = False
        if not wait:
            current_zone = self.zones[current_zone_id]
            target_node = current_zone.nodes[target_node_index]
            was_in_shortage = target_node.is_in_shortage
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
        # check if done
        done = self._episode_is_done()
        if not self.is_single_zone:
            next_zone_state_wrapper = self.get_zone_state()
        else:
            next_zone_state_wrapper = None
        # complete transitions
        for z_s_wrapper, s_wrapper, z_action, action, truck, cost, n_shortages, penalty, z_value, z_log_probs, \
                z_action_mask, value, log_probs, action_mask, policy_index in partial_transitions:
            # truck shortages are the shortages caused by truck action,
            # plus the shortages caused by empty steps, plus the shortages we observe after demand changes
            truck_shortages = self.shortages_since_last_reward + n_shortages
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
            if not self.is_single_zone:
                self.zone_update_current_truck(truck, next_zone_state_wrapper)
            reposition_next_state_wrapper = self.get_reposition_state(z_action, truck.index)
            completed_transitions.append(RawTransition(
                zone_state_wrapper=z_s_wrapper,
                state_wrapper=s_wrapper,
                zone_action=z_action,
                action=action,
                reward=reward,
                zone_next_state_wrapper=next_zone_state_wrapper,
                next_state_wrapper=reposition_next_state_wrapper,
                done=done.to_bool(),
                value=value,
                log_probs=log_probs,
                action_mask=action_mask,
                step_info=self.prepare_step_info(truck_shortages),
                zone_value=z_value,
                zone_log_probs=z_log_probs,
                zone_action_mask=z_action_mask,
                policy_index=policy_index
            ))

        self.shortages_since_last_reward = 0
        self.env_shortages_since_last_reward = 0

        return completed_transitions, done

    def get_reposition_state(
            self,
            zone_action: ZoneAction,
            current_truck_index: int
    ) -> StateWrapper:
        zone_wait, current_zone_id = self.zone_action_values(zone_action)
        state_wrapper = StateWrapper(
            state_types=self.state_types,
            state_builder=self.state_builder,
            current_step=self.current_time_step,
            current_demand=self.current_demand,
            current_zone_id=str(current_zone_id) if not zone_wait else None,
            current_truck_index=current_truck_index,
        )

        if StateFeatureName.PreviousAction in self.state_builder.common_features:
            state_wrapper.set_previous_action(self.state_builder, self.trucks[current_truck_index])
        if StateFeatureName.PreviousFullAction in self.state_builder.common_features:
            state_wrapper.set_previous_full_action(self.state_builder, self.trucks[current_truck_index])

        self.current_state_wrapper = state_wrapper
        return state_wrapper

    def get_zone_state(self) -> StateWrapper:
        state_wrapper = StateWrapper(
            state_types=[self.zone_state_type],
            state_builder=self.state_builder,
            current_step=self.current_time_step,
            current_demand=self.current_demand
        )

        self.current_zone_state_wrapper = state_wrapper
        return state_wrapper

    def apply_reposition_action_on_state(self, action: RepositionAction, truck: Truck) -> StateWrapper:
        return self.apply_action_on_state(action, truck)

    def zone_update_current_truck(
            self,
            current_truck: Truck,
            state_wrapper: StateWrapper,
            previous_truck: Optional[Truck] = None,
            previous_was_wait: Optional[bool] = False,
            previous_reposition_action: Optional[RepositionAction] = None
    ):
        state_wrapper.set_current_truck(current_truck)
        if previous_truck is not None and not previous_was_wait:
            # use previous_truck to set the updated position of the truck used in the previous reposition
            next_step = self.current_time_step.add(Step.from_total_steps(self.time_step.step_size))
            for _, state in state_wrapper.items():
                self.state_builder.apply_zone_action_on_state(
                    state, next_step, previous_truck, previous_reposition_action)
        if StateFeatureName.PreviousZoneAction in self.state_builder.zone_features:
            state_wrapper.set_previous_zone_action(self.state_builder, current_truck)
        if StateFeatureName.PreviousFullAction in self.state_builder.zone_features:
            state_wrapper.set_previous_full_action(self.state_builder, current_truck)

    def unmask_actions(self):
        super().unmask_actions()
        self.unmask_zone_actions()

    def unmask_reposition_actions(self):
        self.action_space.unmask_all()

    def unmask_zone_actions(self):
        self.zone_action_space.unmask_all()

    def zone_action_values(self, action: ZoneAction) -> Tuple[bool, int]:
        return self.zone_action_space.action_to_action_value(action)

    def reposition_action_values(self, action: RepositionAction) -> Tuple[bool, int, int]:
        return self.action_space.action_to_action_value(action)

    def reposition_disable_unavailable_actions(self, zone_action: ZoneAction, current_truck: Truck):
        zone_wait, zone_id = self.zone_action_space.action_to_action_value(zone_action)
        if zone_wait:
            # we disable all reposition actions and we allow only to wait
            self.action_space.disable_all_except_wait()
        else:
            # disable truck unavailable actions
            self.disable_truck_unavailable_actions(current_truck)
            # disable padded nodes
            current_zone = self.zones[str(zone_id)]
            padded_nodes = current_zone.max_zone_size - len(current_zone.nodes) + 1  # because there is also wait action
            for t_action in self.action_space.target_space.get_available_actions()[-padded_nodes:]:
                if not self.action_space.target_space.is_wait_action(t_action):
                    self.action_space.target_space.disable_action(t_action)

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
        state: StateWrapper = self.get_zone_state()
        self.ready = True
        return state

    def store_previous_feature_values(
            self,
            truck: Truck,
            reposition_action: RepositionAction,
            zone_action: ZoneAction,
    ):
        for state_type in self.state_types + [self.zone_state_type]:
            self.state_builder.set_previous_feature(
                state_type, truck, StateFeatureName.PreviousAction, reposition_action)
            self.state_builder.set_previous_feature(
                state_type, truck, StateFeatureName.PreviousZoneAction, zone_action)

    def _reset_env(
            self,
            reset_time_step: bool,
            reset_generator: bool,
            reset_reward: bool,
            reset_nodes: bool,
            reset_trucks: bool,
            show_log: bool
    ):
        super()._reset_env(reset_time_step, reset_generator, reset_reward, reset_nodes, reset_trucks, show_log)
        self.current_zone: Optional[Zone] = None
        self.current_zone_state_wrapper: Optional[StateWrapper] = None
