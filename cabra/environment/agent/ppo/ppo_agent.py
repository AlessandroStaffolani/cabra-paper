from logging import Logger
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import torch as th
from torch import Tensor
from torch.distributions import Categorical

from cabra import SingleRunConfig
from cabra.common.data_structure import RunMode
from cabra.common.distance_helper import Position
from cabra.common.filesystem import create_directory_from_filepath
from cabra.common.math_util import normalize_scalar
from cabra.common.mpi.mpi_pytorch import setup_pytorch_for_mpi
from cabra.common.mpi.mpi_tools import proc_id, num_procs
from cabra.common.stats_tracker import Tracker, last_item
from cabra.core.state import State
from cabra.core.step import Step
from cabra.environment.action_space import RepositionActionSpace, RepositionAction, \
    SubActionSpace, Action
from cabra.environment.agent import AgentType
from cabra.environment.agent.abstract import AgentAbstract, SubActionUtils
from cabra.environment.agent.baseline.constrained import ConstrainedSubActionUtils
from cabra.environment.agent.experience_replay import RolloutBuffer
from cabra.environment.agent.learning_scheduler import LRScheduler
from cabra.environment.agent.ppo.ppo_policy import PPOPolicy
from cabra.environment.config import FullyConnectedLayerConfig, CNNLayerConfig
from cabra.environment.data_structure import StateType, DiscretizeMode, ActionType
from cabra.environment.node import Node, DistancesProvider
from cabra.environment.state_builder import StateBuilder
from cabra.environment.state_wrapper import StateWrapper
from cabra.environment.truck import Truck
from cabra.environment.zone import Zone


class PPOAgent(AgentAbstract):

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
            is_zone_agent: bool,
            distances_provider: DistancesProvider,
            config: Optional[SingleRunConfig] = None,
            mode: RunMode = RunMode.Train,
            init_seed: bool = True,
            random_seed: int = 2,
            **kwargs
    ):
        super(PPOAgent, self).__init__(
            action_space=action_space,
            random_state=random_state,
            name=AgentType.PPO,
            action_spaces=action_spaces,
            state_spaces=state_spaces,
            nodes=nodes,
            zones=zones,
            nodes_max_distance=nodes_max_distance,
            log=log,
            is_zone_agent=is_zone_agent,
            state_builder=state_builder,
            config=config,
            mode=mode,
            **kwargs
        )
        self.rank: int = proc_id()
        self.processes: int = num_procs()
        self.is_root_process: bool = self.rank == 0
        self.use_mpi: bool = self.processes > 1 and self.mode == RunMode.Train
        if self.use_mpi:
            # Special function to avoid certain slowdowns from PyTorch + MPI combo.
            setup_pytorch_for_mpi()
        self.random_seed: int = random_seed
        if init_seed:
            th.manual_seed(self.random_seed)
        # hyperparameters
        ppo_config = self.config.environment.agent.ppo if not self.is_zone_agent \
            else self.config.environment.zones.agent.ppo
        self.lr: float = ppo_config.lr
        self.lr_scheduler_type: LRScheduler = ppo_config.lr_scheduler
        self.n_steps: int = ppo_config.n_steps
        self.local_n_steps: int = int(self.n_steps / num_procs())
        self.batch_size: int = ppo_config.batch_size
        self.n_epochs: int = ppo_config.n_epochs
        self.gamma: float = ppo_config.gamma
        self.gae_lambda: float = ppo_config.gae_lambda
        self.clip_range: float = ppo_config.clip_range
        self.clip_range_vf: Optional[float] = ppo_config.clip_range_vf
        self.normalize_advantage: bool = ppo_config.normalize_advantage
        self.ent_coef: float = ppo_config.ent_coef
        self.vf_coef: float = ppo_config.vf_coef
        self.max_grad_norm: float = ppo_config.max_grad_norm
        self.shared_net_cnn: List[CNNLayerConfig] = ppo_config.shared_net_cnn
        self.shared_net_fcl: List[FullyConnectedLayerConfig] = ppo_config.shared_net_fcl
        self.policy_layers: List[FullyConnectedLayerConfig] = ppo_config.policy_layers
        self.value_layers: List[FullyConnectedLayerConfig] = ppo_config.value_layers
        self.log_std_init: float = ppo_config.log_std_init
        self.prevent_penalties: bool = ppo_config.prevent_penalties
        self.train_steps_with_constraints: int = ppo_config.train_steps_with_constraints
        self.target_kl: Optional[float] = ppo_config.target_kl
        self.deterministic_eval: bool = ppo_config.deterministic_eval
        self.train_frequency: int = ppo_config.train_frequency
        # internal props overriden
        self.requires_evaluation = True
        self.save_agent_state = True
        self.rollout_size = self.local_n_steps
        self.distances_provider: DistancesProvider = distances_provider
        # maybe override sub_action_utils
        self.need_sub_action_utils_switch: bool = False
        # internal props
        self.nodes_positions: np.ndarray = np.array([node.position.to_numpy() for node in self.nodes])
        self.state_type: StateType = StateType.Zone if self.is_zone_agent else StateType.Target
        self.use_continuous: bool = self.config.environment.action_space.use_continuous_action_space
        self.use_wait_space: bool = self.config.environment.action_space.add_wait_space
        self.discretize_mode: DiscretizeMode = self.config.environment.action_space.discretize_mode
        self.target_node_boundaries: Tuple[float, float] = (0, 1)
        # the policy
        self.policy: PPOPolicy = PPOPolicy(
            random_state=self.random,
            input_size=self.state_spaces[self.state_type],
            output_size=self.action_space.distribution_dim_flatten,
            action_space=self.action_space,
            distribution_type=self.action_space.action_space_distribution,
            distribution_dim=self.action_space.distribution_dim,
            config=self.config,
            is_zone_agent=self.is_zone_agent,
            run_mode=self.mode,
            device=self.device,
            init_weights=False if not init_seed and self.mode == RunMode.Eval else True,
            stats_suffix='/zone' if self.is_zone_agent else '/reposition'
        )
        # external props
        self.rollout_buffer: RolloutBuffer = RolloutBuffer(
            capacity=self.local_n_steps,
            state_dim=self.state_spaces[self.state_type],
            action_dim=self.action_space.raw_action_size,
            n_rewards=1,
            random_state=self.random,
            device=self.device,
            gae_lambda=self.gae_lambda,
            gamma=self.gamma,
            use_continuous=self.use_continuous
        )
        # running props
        self.current_latent_pi: Optional[Tensor] = None
        self.current_value: Optional[Tensor] = None
        self.current_log_probs: Optional[Tensor] = None
        self.current_target_node_action: Optional[int] = None
        self.current_action: Optional[np.ndarray] = None
        self.current_original_action: Optional[Tensor] = None

    @property
    def is_deterministic(self) -> bool:
        return self.mode == RunMode.Eval and self.deterministic_eval

    def reset_current_action_props(self):
        self.current_latent_pi: Optional[Tensor] = None
        self.current_value: Optional[Tensor] = None
        self.current_log_probs: Optional[Tensor] = None
        self.current_target_node_action: Optional[int] = None
        self.current_original_action: Optional[Tensor] = None

    def choose_full_action(self, state: State):
        self.reset_current_action_props()

        action, value, log_probs = self.policy.get_full_action(state, deterministic=self.is_deterministic)
        self.current_value = value
        self.current_action = action.squeeze(0)
        self.current_original_action = action
        self.current_log_probs = log_probs

    def choose_wait_action(self, state: State) -> int:
        if self.prevent_penalties:
            # Ensure usage of fresh values for these props
            self.reset_current_action_props()
            latent_pi, value = self.policy.prepare_latent(state)
            self.current_latent_pi = latent_pi
            self.current_value = value

            # at this point the mask of the first sub-action has already masked if necessary
            # index 0 represents the index of the mask and the distribution associated to the target_node sub-action
            wait_action = self.policy.get_action(
                index=ActionType.Wait, latent_pi=latent_pi, deterministic=self.is_deterministic)
            if self.use_continuous:
                self.current_original_action = th.tensor([wait_action, 0, 0, 0], dtype=th.float, device=self.device)
            else:
                self.current_original_action = th.tensor([wait_action, 0, 0], dtype=th.float, device=self.device)

            return wait_action
        else:
            self.choose_full_action(state)
            assert self.current_action is not None
            return self.current_action[0]

    def choose_target_node(
            self,
            state: State,
            t: Step,
            epsilon: float,
            random: float,
            truck: Truck,
            current_zone_id: str,
    ) -> int:
        if self.prevent_penalties:

            if not self.use_wait_space:
                # Ensure usage of fresh values for these props
                self.reset_current_action_props()
                latent_pi, value = self.policy.prepare_latent(state)
                self.current_latent_pi = latent_pi
                self.current_value = value
            assert self.current_latent_pi is not None

            target_node_action = self.policy.get_action(
                index=ActionType.Target, latent_pi=self.current_latent_pi, deterministic=self.is_deterministic)
            if self.use_continuous:
                target_node_action = target_node_action.squeeze(0)
            if not self.use_wait_space:
                if self.use_continuous:
                    self.current_original_action = th.concatenate([
                        th.tensor(target_node_action, dtype=th.float, device=self.device),
                        th.tensor([0], dtype=th.float, device=self.device)
                    ]).to(self.device)
                else:
                    self.current_original_action = th.tensor(
                        [target_node_action, 0], dtype=th.float, device=self.device)
            else:
                if self.use_continuous:
                    self.current_original_action[1:3] = th.tensor(target_node_action)
                else:
                    self.current_original_action[1] = target_node_action

            # if use_continuous get the actual node index
            if self.use_continuous:
                target_node_action = self._get_node_from_coordinates(target_node_action)
            self.current_target_node_action = target_node_action

            return target_node_action
        else:
            assert self.current_action is not None
            return self.current_action[1]

    def choose_quantity(
            self,
            state: State,
            t: Step,
            epsilon: float,
            random: float,
            truck: Truck,
            current_zone_id: str,
    ) -> int:
        if self.prevent_penalties:
            assert self.current_latent_pi is not None

            # at this point the mask of the second sub-action has already masked if necessary
            # index 1 represents the index of the mask and the distribution associated to the quantity sub-action
            quantity_action = self.policy.get_action(
                index=ActionType.Quantity, latent_pi=self.current_latent_pi, deterministic=self.is_deterministic)
            self.current_original_action[-1] = quantity_action
            action_tensor = self.current_original_action.to(self.device).unsqueeze(0)
            self.current_log_probs = self.policy.get_log_probs(latent_pi=self.current_latent_pi, action=action_tensor)

            return quantity_action
        else:
            assert self.current_action is not None
            return self.current_action[2]

    def _choose_action(
            self,
            state_wrapper: StateWrapper,
            t: Step,
            truck: Truck,
            current_zone_id,
            epsilon: float,
            random: float,
            **kwargs
    ) -> RepositionAction:
        state = state_wrapper.get_state(self.state_type)
        # choose the wait action
        w_a = None
        if self.use_wait_space:
            w_a = self.choose_wait_action(state)
        # prepare target_node sub-action
        self.sub_actions_utils.prepare_target_node_sub_action(truck, current_zone_id)
        self.apply_wait_action_to_sub_action_space(wait_action=w_a, sub_action_space=self.action_space.target_space)
        # choose the target node
        t_node_a = self.choose_target_node(state, t, epsilon, random, truck, current_zone_id)
        # prepare the quantity sub-action
        self.sub_actions_utils.prepare_quantity_sub_action(t_node_a, truck, state_wrapper, current_zone_id)
        self.apply_wait_action_to_sub_action_space(wait_action=w_a, sub_action_space=self.action_space.quantity_space)
        # chose quantity
        q_a = self.choose_quantity(state, t, epsilon, random, truck, current_zone_id)

        # self.count_available_action(RepositionAction(wait=w_a, target=t_node_a, quantity=q_a), t)

        return RepositionAction(wait=w_a, target=t_node_a, quantity=q_a)

    def count_available_action(self, action: RepositionAction, step: Step):
        action_index = self.choose_action_calls
        is_wait_action = self.action_space.is_wait_action(action)
        nodes_selectable = 0
        if (len(self.action_space.target_space.get_available_actions()) > 1
                or self.action_space.target_space.get_available_actions()[0] !=
                self.action_space.target_space.wait_action_index):
            nodes_selectable = len(self.action_space.target_space.get_available_actions())
        filename = (f'data/results/available_nodes/'
                    f'{self.config.emulator.model.cdrc_data_model.eval_dataset_path.replace("datasets/cdrc/", "").replace(".json", "").replace("/", "-")}/{self.mode.value}_dataset.csv')
        create_directory_from_filepath(filename)
        if action_index == 0:
            with open(filename, 'w') as f:
                f.write('mode,index,selectable_nodes,is_wait_action,step,weekday,week,month\n')
        with open(filename, 'a') as f:
            f.write(f'{self.mode.value},'
                    f'{action_index},'
                    f'{nodes_selectable},'
                    f'{1 if is_wait_action is True else 0},'
                    f'{step.to_str()},'
                    f'{step.week_day},'
                    f'{step.week},'
                    f'{step.month}\n')

    def apply_wait_action_to_sub_action_space(self, wait_action: Optional[int], sub_action_space: SubActionSpace):
        if wait_action is not None:
            if bool(wait_action):
                # it is wait
                sub_action_space.disable_all_except_wait()
            else:
                # it is not wait
                sub_action_space.disable_wait_action()
                if len(sub_action_space.get_available_actions()) == 0:
                    sub_action_space.enable_wait_action()

    def choose_zone_action(self, state: State, t: Step, epsilon: float, random: float, truck: Truck) -> int:
        self.reset_current_action_props()

        action, value, log_probs = self.policy.get_full_action(
            state, mask=self.action_space.zone_space.get_mask(), deterministic=self.is_deterministic)
        self.current_value = value
        self.current_original_action = action
        self.current_log_probs = log_probs

        return action.item()

    def step_action_info(self, action_info: Dict[str, Any]):
        assert self.current_value is not None and self.current_log_probs is not None
        action_info['value'] = self.current_value
        action_info['log_probs'] = self.current_log_probs
        action_info['original_action'] = self.current_original_action
        action_info['action_mask'] = self.action_space.get_mask() if self.prevent_penalties else None

    def _get_node_from_coordinates(self, target_node_coordinates: np.ndarray) -> int:
        available_actions = self.action_space.target_space.get_available_actions()
        if len(available_actions) == 1 and self.action_space.target_space.is_wait_action(available_actions[0]):
            # we can only wait, we return wait
            return self.action_space.target_space.wait_action_index
        clipped_target = np.clip(target_node_coordinates, *self.target_node_boundaries)
        target_lng = clipped_target[0]
        target_lat = clipped_target[1]
        target_lng = normalize_scalar(
            target_lng,
            min_val=self.target_node_boundaries[0],
            max_val=self.target_node_boundaries[1],
            a=self.distances_provider.min_position.lng,
            b=self.distances_provider.max_position.lng
        )
        target_lat = normalize_scalar(
            target_lat,
            min_val=self.target_node_boundaries[0],
            max_val=self.target_node_boundaries[1],
            a=self.distances_provider.min_position.lat,
            b=self.distances_provider.max_position.lat
        )
        nodes_mask = self.action_space.target_space.get_mask()[:-1]  # last one is wait action
        distances = DistancesProvider.point_distance_from_nodes(
            Position(lat=target_lat, lng=target_lng), self.nodes_positions)

        distances = 1 - distances
        distances = normalize_scalar(distances, min_val=distances.min(), max_val=distances.max(), a=0, b=1)
        distances[nodes_mask] = float(np.finfo(np.float32).min)
        if self.is_deterministic or self.discretize_mode == DiscretizeMode.Greedy:
            return distances.argmax()
        else:
            dist = Categorical(logits=th.from_numpy(distances))
            sampled = dist.sample().item()
            return sampled

    def learn(self):
        self.policy.set_mode(RunMode.Train)
        self._update_current_progress_remaining()
        if self.learn_calls % self.train_frequency == 0 and self.rollout_buffer.is_full:
            self.continue_training = self.policy.train(self.rollout_buffer)
        self.learn_calls += 1

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
            **kwargs
    ):
        self.rollout_buffer.push(
            state=state_wrapper.get_state(self.state_type),
            action=action,
            reward=reward,
            next_state=None,
            done=done,
            value=value,
            log_probs=log_probs,
            action_mask=action_mask)

    def is_buffer_full(self, offset: int = 0) -> bool:
        return self.rollout_buffer.will_be_full(offset)

    def evaluate_state_value(self, state_wrapper: StateWrapper) -> Optional[Tensor]:
        return self.policy.evaluate_state(state_wrapper.get_state(self.state_type))

    def start_rollout(self):
        self.rollout_buffer.reset()
        # During the rollout collection we set to eval only the policy, so batch norm and dropout are not utilized.
        # The train method will set back to training modo
        self.policy.set_mode(RunMode.Eval)

    def end_rollout(self, last_value: Tensor, last_done: bool):
        self.rollout_buffer.compute_returns_and_advantage(last_value, last_done)

    def set_stats_tracker(self, tracker: Tracker):
        self.stats_tracker: Tracker = tracker
        self.policy.set_stats_tracker(tracker)
        self.init_extra_tracked_variables()

    def get_model(self, net_type: StateType = None, policy_net=True) -> Optional[th.nn.Module]:
        return self.policy.policy_net

    def get_agent_state(self) -> Dict[str, Any]:
        agent_state = super(PPOAgent, self).get_agent_state()
        agent_state['policy'] = self.policy.get_model_state()
        return agent_state

    def load_agent_state(self, agent_state: Dict[str, Any]):
        super(PPOAgent, self).load_agent_state(agent_state)
        self.policy.load_model_state(agent_state['policy'])

    def set_mode(self, mode: RunMode):
        super(PPOAgent, self).set_mode(mode)
        self.policy.set_mode(mode)

    def init_extra_tracked_variables(self):
        prefix = RunMode.Train.value
        metrics = ['loss', 'policy_loss', 'value_loss', 'entropy_loss', 'approx_kl_divergence', 'lr']
        suffix = 'zone' if self.is_zone_agent else 'reposition'
        for m in metrics:
            self.stats_tracker.init_tracking(f'{prefix}/{m}/{suffix}', tensorboard=True, redis_save=True,
                                             aggregation_fn=last_item, str_precision=8)

    def get_tracked_learning_params(self) -> Dict[str, str]:
        if not self.is_single_zone or not self.is_zone_agent:
            prefix = RunMode.Train.value
            metrics = ['loss', 'policy_loss', 'value_loss', 'entropy_loss', 'approx_kl_divergence', 'lr']
            suffix = 'zone' if self.is_zone_agent else 'reposition'
            learning_params = {}
            for m in metrics:
                full_key = f'{prefix}/{m}/{suffix}'
                short_key = f'{m}/{suffix}'
                learning_params[short_key] = self.stats_tracker.get_key_to_string(full_key, aggr_fn=last_item)
            return learning_params
        else:
            return {}

    def _update_current_progress_remaining(self) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)
        """
        total_steps = self.config.run.training_steps
        self.policy.current_progress_remaining = 1.0 - float(self.learn_calls) / float(total_steps)


class ConstrainedPPOAgent(PPOAgent):

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
            is_zone_agent: bool,
            distances_provider: DistancesProvider,
            config: Optional[SingleRunConfig] = None,
            mode: RunMode = RunMode.Train,
            init_seed: bool = True,
            random_seed: int = 2,
            **kwargs,
    ):
        super().__init__(action_space, random_state, action_spaces, state_spaces, nodes, zones, nodes_max_distance, log,
                         state_builder, is_zone_agent, distances_provider, config,
                         mode, init_seed, random_seed, **kwargs)
        self.name = AgentType.ConstrainedPPO
        ppo_config = self.config.environment.agent.ppo if not self.is_zone_agent \
            else self.config.environment.zones.agent.ppo
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


class PPOSubActionConstrains(SubActionUtils):

    def __init__(
            self,
            action_space: RepositionActionSpace,
            nodes: List[Node],
            zones: Dict[str, Zone],
            config: SingleRunConfig,
            nodes_max_distance: float,
            state_builder: StateBuilder,
            distances_provider: DistancesProvider,
            force_action_swap: bool
    ):
        super().__init__(action_space, nodes, zones, config, nodes_max_distance, state_builder)
        self.distances_provider: DistancesProvider = distances_provider
        self.critical_threshold: float = self.nodes[0].critical_threshold
        self.force_action_swap: bool = force_action_swap

        self.next_is_pick_mapping: Dict[int, bool] = {i: True for i in range(self.config.environment.trucks.n_trucks)}

    def prepare_target_node_sub_action(self, truck: Truck, current_zone_id: Optional[str]):
        all_disabled = True
        next_is_pick = self.next_is_pick_mapping[truck.index]
        stats = {'critically_empty_nodes': 0, 'critically_full_nodes': 0, 'nodes_fullness_ratio': {}}
        if current_zone_id is not None:
            # we can disable actions only if current_zone_id is not None,
            # otherwise it means zone_action is wait, and we can only. Only wait is already set by the environment
            zone_nodes = self.zones[current_zone_id].nodes
            for node_index, node in enumerate(zone_nodes):
                action_index = self.action_space.target_space.inverted_actions_mapping[node_index]
                got_enabled = False
                if node.is_full_critical() and (next_is_pick or not self.force_action_swap):
                    # node has many bikes, enable this
                    self.action_space.target_space.enable_action(action_index)
                    all_disabled = False
                    got_enabled = True
                    stats['critically_full_nodes'] += 1
                else:
                    # node has more empty slots than threshold, disable this
                    self.action_space.target_space.disable_action(action_index)
                if node.is_empty_critical() and (not next_is_pick or not self.force_action_swap):
                    # node has few bikes, enable this
                    self.action_space.target_space.enable_action(action_index)
                    all_disabled = False
                    stats['critically_empty_nodes'] += 1
                else:
                    # node has more bikes than threshold, disable this
                    if not got_enabled:
                        self.action_space.target_space.disable_action(action_index)
                stats['nodes_fullness_ratio'][node.fullness_ratio()] = node_index
            # # empty to full swap required?
            # if truck.load == 0 and stats['critically_empty_nodes'] > 0 and stats['critically_full_nodes'] == 0:
            #     self.empty_to_full_rule_swap(stats['critically_empty_nodes'], stats['nodes_fullness_ratio'], truck)
            # elif truck.empty_slots == 0 and stats['critically_full_nodes'] > 0 and stats['critically_empty_nodes'] == 0:
            #     # full to empty swap required?
            #     self.full_to_empty_rule_swap(stats['critically_full_nodes'], stats['nodes_fullness_ratio'], truck)

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

    def full_to_empty_rule_swap(self, nodes_to_enable: int, nodes_fullness_ratio: Dict[float, int], truck: Truck):
        self.action_space.target_space.disable_all()
        indexes_to_enable = [
            nodes_fullness_ratio[key] for key in sorted(nodes_fullness_ratio.keys(), reverse=False)[:nodes_to_enable]]
        for node_index in indexes_to_enable:
            self.action_space.target_space.enable_action(node_index)
        # quantity sub-action must be a drop now
        self.next_is_pick_mapping[truck.index] = False

    def target_node_requires_pick_action(
            self,
            target_node_action: int,
            truck: Truck,
            current_zone_id: Optional[str],
    ) -> bool:
        if not self.action_space.target_space.is_wait_action(target_node_action):
            zone_nodes = self.zones[current_zone_id].nodes
            target_node: Node = zone_nodes[target_node_action]
            if target_node.is_full_critical():
                # node is full, we need to pick here
                return True
            elif target_node.is_empty_critical():
                # node is empty, we need to drop here
                return False
        return self.next_is_pick_mapping[truck.index]

    def prepare_quantity_sub_action(
            self,
            target_node_action: int,
            truck: Truck,
            state_wrapper: StateWrapper,
            current_zone_id: Optional[str],
    ):
        next_is_pick = self.target_node_requires_pick_action(target_node_action, truck, current_zone_id)
        if current_zone_id is not None:
            # we can disable actions only if current_zone_id is not None,
            # otherwise it means zone_action is wait, and we can only. Only wait is already set by the environment
            zone_nodes = self.zones[current_zone_id].nodes

            if not self.action_space.target_space.is_wait_action(target_node_action):
                target_node: Node = zone_nodes[target_node_action]
                for q_action in self.action_space.quantity_space.get_available_actions():
                    if not self.action_space.quantity_space.is_wait_action(q_action):
                        q_value = self.action_space.quantity_space.actions_mapping[q_action]
                        if q_value > 0:
                            # pick action
                            if not truck.pick_possible(target_node, q_value) or not next_is_pick:
                                self.action_space.quantity_space.disable_action(q_action)
                            else:
                                self.action_space.quantity_space.enable_action(q_action)
                        else:
                            # drop action
                            if not truck.drop_possible(target_node, q_value) or next_is_pick:
                                self.action_space.quantity_space.disable_action(q_action)
                            else:
                                self.action_space.quantity_space.enable_action(q_action)
                if len(self.action_space.quantity_space.get_available_actions()) == 1:
                    self.action_space.quantity_space.enable_wait_action()
                else:
                    self.action_space.quantity_space.disable_wait_action()
            else:
                # is wait action, we can only wait
                self.action_space.quantity_space.disable_all_except_wait()

        # we swap this variable
        self.next_is_pick_mapping[truck.index] = not next_is_pick
