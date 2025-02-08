from typing import Optional, Tuple, Any, List, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from numpy import ndarray
from numpy.random import RandomState
from torch import Tensor

from cabra import SingleRunConfig
from cabra.common.data_structure import RunMode
from cabra.common.mpi.mpi_pytorch import sync_params, mpi_avg_grads
from cabra.common.mpi.mpi_tools import proc_id, num_procs
from cabra.common.stats_tracker import Tracker
from cabra.core.state import State
from cabra.environment.action_space import RepositionActionSpace
from cabra.environment.data_structure import DistributionType, ActionType
from cabra.environment.agent.experience_replay import RolloutBuffer
from cabra.environment.agent.learning_scheduler import LRScheduler, Schedule, get_scheduler
from cabra.environment.agent.ppo.actor_critic_model import ActorCriticModel
from cabra.environment.agent.torch_utils import update_learning_rate
from cabra.environment.config import CNNLayerConfig, FullyConnectedLayerConfig


class PPOPolicy:

    def __init__(
            self,
            random_state: RandomState,
            input_size: int,
            output_size: int,
            action_space: RepositionActionSpace,
            config: SingleRunConfig,
            distribution_type: DistributionType,
            distribution_dim: int,
            is_zone_agent: bool,
            run_mode: RunMode = RunMode.Train,
            device: th.device = th.device('cpu'),
            init_weights: bool = True,
            stats_suffix: str = '',
            share_grads: bool = True
    ):
        # external props
        self.rank: int = proc_id()
        self.processes: int = num_procs()
        self.is_root_process: bool = self.rank == 0
        self.use_mpi: bool = self.processes > 1 and run_mode == RunMode.Train
        self.config: SingleRunConfig = config
        self.random_state: RandomState = random_state
        self.action_space: RepositionActionSpace = action_space
        self.input_size: int = input_size
        self.output_size: int = output_size
        self.run_mode: str = run_mode
        self.device: th.device = device
        self.stats_tracker: Optional[Tracker] = None
        self.current_progress_remaining: float = 1
        self.is_zone_agent: bool = is_zone_agent
        self.share_grads: bool = share_grads
        # PPO Hyperparameters
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
        self.target_kl: Optional[float] = ppo_config.target_kl
        self.distribution_type: DistributionType = distribution_type
        self.distribution_dim: int = distribution_dim

        # model props
        self.lr_scheduler: Schedule = get_scheduler(self.lr_scheduler_type, self.lr)
        self.policy_net: ActorCriticModel = ActorCriticModel(
            input_size=input_size,
            output_size=output_size,
            shared_cnn_units=self.shared_net_cnn,
            shared_fcl_units=self.shared_net_fcl,
            policy_layers=self.policy_layers,
            value_layers=self.value_layers,
            distribution_type=self.distribution_type,
            distribution_dim=distribution_dim,
            action_type_mapping=self.action_space.action_type_mapping,
            log_std_init=self.log_std_init,
            init_weights=init_weights
        ).to(self.device)
        self.optimizer = th.optim.Adam(self.policy_net.parameters(), lr=self.lr_scheduler(1))

        # Internal props
        self.stats_suffix: str = stats_suffix
        self.train_steps: int = 0
        self.use_cnn: bool = len(self.shared_net_cnn) > 0
        self.use_continuous: bool = self.config.environment.action_space.use_continuous_action_space

        # sync parallel workers if MPI enabled
        if self.use_mpi:
            sync_params(self.policy_net)

    def set_stats_tracker(self, tracker: Tracker):
        self.stats_tracker: Tracker = tracker

    def _preprocess_state(self, state: State) -> Tensor:
        # unsqueeze(0) to bring the state in batch shape
        if self.use_cnn:
            state_tensor = state.to_matrix_tensor(device=self.device).unsqueeze(0).unsqueeze(0)
        else:
            state_tensor = state.to_tensor(device=self.device).unsqueeze(0)
        return state_tensor

    @th.no_grad()
    def get_full_action(
            self,
            state: State,
            deterministic: bool = False,
            mask: Optional[ndarray] = None,
            **kwargs
    ) -> Tuple[ndarray, Tensor, Tensor]:
        # get actions, values and log probabilities of the actions
        actions, values, log_prob = self.policy_net(self._preprocess_state(state), deterministic, mask)

        return actions.cpu().numpy(), values, log_prob

    @th.no_grad()
    def get_action(self, index: Union[int, ActionType], latent_pi: Tensor, deterministic: bool = False) -> Union[
        int, np.ndarray]:
        action = self.policy_net.get_action_from_distribution(
            latent_pi,
            index,
            mask=self.action_space.get_mask(),
            deterministic=deterministic
        )
        if len(action.size()) > 1 and action.size()[1] > 1:
            return action.cpu().numpy()
        return action.item()

    @th.no_grad()
    def prepare_latent(self, state: State) -> Tuple[Tensor, Tensor]:
        return self.policy_net.get_latent_pi_and_value(self._preprocess_state(state))

    @th.no_grad()
    def get_log_probs(self, latent_pi: Tensor, action: Tensor) -> Tensor:
        return self.policy_net.get_log_prob(latent_pi, action, mask=[self.action_space.get_mask()])

    @th.no_grad()
    def evaluate_state(self, state: State) -> Tensor:
        value = self.policy_net.get_value(self._preprocess_state(state))
        return value

    def train(self, rollout_buffer: RolloutBuffer, *args, **kwargs) -> bool:
        # Switch to train mode (this affects batch norm / dropout)
        self.set_mode(RunMode.Train)
        # Update optimizer learning rate
        self._update_learning_rate(self.optimizer)

        losses = []
        entropy_losses = []
        pg_losses, value_losses = [], []
        approx_kl_divs = []

        iteration_train_steps = 0
        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            # Do a complete pass on the rollout buffer
            samples = rollout_buffer.sample(self.batch_size, device=self.device)
            for (states, actions, old_values, old_log_prob, advantages, returns, action_masks) in samples:
                states = states.float()
                if self.use_cnn:
                    states = states.unsqueeze(1)
                if self.distribution_type == DistributionType.Categorical:
                    # Convert discrete action from float to long
                    actions = actions.long().flatten()
                values, log_prob, entropy = self.policy_net.evaluate_actions(states, actions, action_masks)
                values = values.flatten()
                # Normalize advantage
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
                pg_losses.append(policy_loss.item())

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = old_values + th.clamp(values - old_values, -self.clip_range_vf, self.clip_range_vf)
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(-entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss
                losses.append(loss.item())

                # Calculate approximate form of reverse KL Divergence for early stopping
                with th.no_grad():
                    log_ratio = log_prob - old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
                # average grads between workers
                if self.use_mpi and self.share_grads:
                    mpi_avg_grads(self.policy_net)
                self.optimizer.step()
                # increase the training steps counter
                iteration_train_steps += 1

            if not continue_training:
                break

        self.train_steps += iteration_train_steps

        # track training variables
        prefix = RunMode.Train.value
        self.stats_tracker.track(f'{prefix}/loss{self.stats_suffix}', np.mean(losses).item())
        self.stats_tracker.track(f'{prefix}/policy_loss{self.stats_suffix}', np.mean(pg_losses).item())
        self.stats_tracker.track(f'{prefix}/value_loss{self.stats_suffix}', np.mean(value_losses).item())
        self.stats_tracker.track(f'{prefix}/entropy_loss{self.stats_suffix}', np.mean(entropy_losses).item())
        self.stats_tracker.track(f'{prefix}/approx_kl_divergence{self.stats_suffix}', np.mean(approx_kl_divs).item())
        self.stats_tracker.track(f'{prefix}/training_steps{self.stats_suffix}', iteration_train_steps)
        if self.is_zone_agent:
            self.stats_tracker.track(f'{prefix}/training_steps', iteration_train_steps)

        return continue_training

    def get_model_state(self):
        return {
            'policy_net': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_model_state(self, model_state: dict):
        self.policy_net.load_state_dict(model_state['policy_net'])
        self.optimizer.load_state_dict(model_state['optimizer'])

    def set_mode(self, mode: RunMode):
        self.run_mode = mode
        if mode == RunMode.Train:
            self.policy_net.train()
        else:
            self.policy_net.eval()

    def _update_learning_rate(self, optimizer: th.optim.Optimizer):
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).
        """
        current_lr = self.lr_scheduler(self.current_progress_remaining)
        update_learning_rate(optimizer, current_lr)
        # Log the current learning rate
        prefix = RunMode.Train.value
        self.stats_tracker.track(f'{prefix}/lr{self.stats_suffix}', current_lr)
