from typing import Optional, Union, Generator, List, Tuple

import numpy as np
import torch
import torch as th
from numpy import ndarray
from torch import Tensor

from cabra.core.state import State
from cabra.environment.action_space import Action
from cabra.environment.agent.experience_replay.base_buffer import BaseBuffer
from cabra.environment.agent.experience_replay.experience_entry import TransitionEntry, \
    RolloutEntry


class RolloutBuffer(BaseBuffer):

    def __init__(
            self,
            capacity: int,
            state_dim: int,
            action_dim: int = 2,
            n_rewards: int = 1,
            random_state: np.random.RandomState = None,
            device: th.device = th.device('cpu'),
            gae_lambda: float = 1,
            gamma: float = 0.99,
            use_continuous: bool = False,
            **kwargs
    ):
        super(RolloutBuffer, self).__init__(capacity, random_state, TransitionEntry.__name__, device)
        self.capacity = capacity
        self.state_dim: int = state_dim
        self.action_dim: int = action_dim
        self.n_rewards: int = n_rewards
        self.gae_lambda: float = gae_lambda
        self.gamma: float = gamma
        self.use_continuous: bool = use_continuous

        self.states: ndarray = np.zeros((self.capacity, self.state_dim), dtype=np.float)
        self.actions: ndarray = np.zeros((self.capacity, self.action_dim),
                                         dtype=np.long if not self.use_continuous else np.float)
        self.rewards: ndarray = np.zeros((self.capacity, self.n_rewards), dtype=np.float)
        self.dones: ndarray = np.zeros((self.capacity, 1), dtype=np.bool)
        self.log_probs: ndarray = np.zeros((self.capacity, 1), dtype=np.float)
        self.values: ndarray = np.zeros((self.capacity, 1), dtype=np.float)
        self.advantages: ndarray = np.zeros((self.capacity, 1), dtype=np.float)
        self.returns: ndarray = np.zeros((self.capacity, 1), dtype=np.float)
        self.action_masks: List[Tuple[np.ndarray, ...]] = [tuple()] * self.capacity

        self.count: int = 0
        self.is_full: bool = False

    def __str__(self):
        return f'<RolloutBuffer capacity={self.capacity} size={len(self)} >'

    def __len__(self):
        return self.count

    def reset(self):
        self.states: ndarray = np.zeros((self.capacity, self.state_dim), dtype=np.float)
        self.actions: ndarray = np.zeros((self.capacity, self.action_dim),
                                         dtype=np.long if not self.use_continuous else np.float)
        self.rewards: ndarray = np.zeros((self.capacity, self.n_rewards), dtype=np.float)
        self.dones: ndarray = np.zeros((self.capacity, 1), dtype=np.bool)
        self.log_probs: ndarray = np.zeros((self.capacity, 1), dtype=np.float)
        self.values: ndarray = np.zeros((self.capacity, 1), dtype=np.float)
        self.advantages: ndarray = np.zeros((self.capacity, 1), dtype=np.float)
        self.returns: ndarray = np.zeros((self.capacity, 1), dtype=np.float)
        self.action_masks: List[Tuple[np.ndarray, ...]] = [tuple()] * self.capacity

        self.count: int = 0
        self.is_full: bool = False

    def push(
            self,
            state: State,
            action: Union[int, Action],
            reward: float,
            next_state: Optional[State],
            done: bool,
            value: Optional[Tensor] = None,
            log_probs: Optional[Tensor] = None,
            action_mask: Optional[Tuple[np.ndarray, ...]] = None,
            matrix_state: bool = False,
    ):
        action = action if isinstance(action, torch.Tensor) else action.to_tensor()
        entry = TransitionEntry(**self._push_pre_processing_numpy(
            state, action, reward, done, value, log_probs, action_mask, matrix_state))

        self.states[self.count] = entry.state
        self.actions[self.count] = entry.action
        self.rewards[self.count] = entry.reward
        self.dones[self.count] = entry.done
        self.values[self.count] = entry.value
        self.log_probs[self.count] = entry.log_probs
        self.action_masks[self.count] = entry.action_mask

        self.count += 1
        if len(self) == self.capacity:
            self.is_full = True

    def compute_returns_and_advantage(self, last_values: Tensor, last_done: bool) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))
        """
        # Reshape
        last_values = last_values.squeeze(0).detach().cpu().numpy()

        last_gae_lam = 0
        for step in reversed(range(self.capacity)):
            if step == self.capacity - 1:
                next_non_terminal = 1.0 - last_done
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            try:
                last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            except FloatingPointError:
                last_gae_lam = last_gae_lam
            self.advantages[step] = last_gae_lam
        self.returns = self.advantages + self.values

    def sample(self, batch_size: Optional[int], *args, **kwargs) -> Generator[RolloutEntry, None, None]:
        if batch_size is None:
            batch_size = self.capacity
        indices = self.random.permutation(self.capacity)

        start_idx = 0
        while start_idx < self.capacity:
            sampled_indexes = indices[start_idx: start_idx + batch_size]

            if self.action_masks[0] is None:
                sample_masks = None
            else:
                sample_masks = [self.action_masks[i] for i in sampled_indexes]

            yield RolloutEntry(
                states=self.to_tensor(self.states[sampled_indexes]),
                actions=self.to_tensor(self.actions[sampled_indexes],
                                       dtype=th.long if not self.use_continuous else th.float),
                values=self.to_tensor(self.values[sampled_indexes].squeeze(-1)),
                log_probs=self.to_tensor(self.log_probs[sampled_indexes].squeeze(-1)),
                advantages=self.to_tensor(self.advantages[sampled_indexes].squeeze(-1)),
                returns=self.to_tensor(self.returns[sampled_indexes].squeeze(-1)),
                action_masks=sample_masks
            )
            start_idx += batch_size

    def to_tensor(self, array: ndarray, dtype=th.float) -> Tensor:
        return th.as_tensor(array, dtype=dtype, device=self.device)

    def will_be_full(self, offset: int) -> bool:
        return self.count + offset >= self.capacity
