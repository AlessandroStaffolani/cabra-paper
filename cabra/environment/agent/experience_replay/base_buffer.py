from abc import abstractmethod
from collections import deque
from typing import Union, Optional, Deque, Any, Dict, Tuple

import numpy as np
import torch
from torch import Tensor

from cabra.core.state import State
from cabra.environment.agent.experience_replay.experience_entry import ExperienceEntry, TransitionEntry
from cabra.common.data_structure import BaseEntry


class BaseBuffer:

    def __init__(
            self,
            capacity: int,
            random_state: np.random.RandomState = None,
            memory_entry: str = ExperienceEntry.__name__,
            device: torch.device = torch.device('cpu'),
    ):
        if random_state is not None:
            self.random: np.random.RandomState = random_state
        else:
            self.random: np.random.RandomState = np.random.RandomState()
        self.capacity: int = capacity
        self.memory: Deque[BaseEntry] = deque(maxlen=self.capacity)
        self.memory_entry: str = memory_entry
        self.device: torch.device = device

    def __len__(self):
        return len(self.memory)

    @abstractmethod
    def push(
            self,
            state: State,
            action: int,
            reward: float,
            next_state: State,
            done: bool,
            value: Optional[Tensor] = None,
            log_probs: Optional[Tensor] = None,
            action_mask: Optional[Tuple[np.ndarray, ...]] = None,
            matrix_state: bool = False,
    ):
        pass

    @abstractmethod
    def sample(self, batch_size: int, *args, **kwargs) -> BaseEntry:
        pass

    def _push_pre_processing(
            self,
            state: State,
            action: Union[int, Tensor],
            reward: float,
            next_state: State,
            done: bool,
            value: Optional[Tensor] = None,
            log_probs: Optional[Tensor] = None,
            action_mask: Optional[Tuple[np.ndarray, ...]] = None,
            matrix_state: bool = False
    ) -> Dict[str, Any]:
        if isinstance(action, int):
            action_to_push = torch.tensor([action], dtype=torch.long, device=self.device)
        else:
            action_to_push = action
        if matrix_state:
            state_tensor = state.to_matrix_tensor(self.device).unsqueeze(0)
            next_state_tensor = next_state.to_matrix_tensor(self.device).unsqueeze(0)
        else:
            state_tensor = state.to_tensor(self.device)
            next_state_tensor = next_state.to_tensor(self.device)

        fields = {
            'state': state_tensor,
            'action': action_to_push,
            'reward': torch.tensor([reward], dtype=torch.float, device=self.device),
            'next_state': next_state_tensor,
            'done': torch.tensor([done], dtype=torch.float, device=self.device)
        }

        if self.memory_entry == ExperienceEntry.__name__:
            return fields
        elif self.memory_entry == TransitionEntry.__name__:
            fields['value'] = value
            fields['log_probs'] = log_probs
            fields['action_mask'] = action_mask
            return fields

    def _push_pre_processing_numpy(
            self,
            state: State,
            action: Union[int, Tensor],
            reward: float,
            done: bool,
            value: Optional[Tensor] = None,
            log_probs: Optional[Tensor] = None,
            action_mask: Optional[Tuple[np.ndarray, ...]] = None,
            matrix_state: bool = False
    ) -> Dict[str, Any]:
        if isinstance(action, int):
            action_to_push = np.array([action], dtype=np.long)
        else:
            action_to_push = action.cpu().numpy()
        if matrix_state:
            state_tensor = state.to_matrix_tensor(self.device).unsqueeze(0).cpu().numpy()
        else:
            state_tensor = state.to_tensor(self.device).cpu().numpy()

        fields = {
            'state': state_tensor,
            'action': action_to_push,
            'reward': np.array([reward], dtype=np.float),
            'next_state': None,
            'done': np.array([done], dtype=np.float)
        }

        if self.memory_entry == ExperienceEntry.__name__:
            return fields
        elif self.memory_entry == TransitionEntry.__name__:
            fields['value'] = value.cpu().numpy()
            fields['log_probs'] = log_probs.cpu().numpy()
            fields['action_mask'] = action_mask
            return fields
