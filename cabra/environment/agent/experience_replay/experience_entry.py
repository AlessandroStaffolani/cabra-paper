from dataclasses import dataclass, field
from typing import Union, List, Optional, Dict, Any, Tuple

import numpy as np
from torch import Tensor

from cabra.common.data_structure import BaseEntry
from cabra.environment.action_space import RepositionAction, ZoneAction
from cabra.environment.state_wrapper import StateWrapper


@dataclass
class ExperienceEntry(BaseEntry):
    state: Union[Tensor, List[Tensor]]
    action: Union[Tensor, List[Tensor]]
    reward: Union[Tensor, List[Tensor]]
    next_state: Union[Tensor, List[Tensor]]
    done: Optional[Union[Tensor, List[Tensor]]] = field(default=None, init=True)
    weights: Optional[Tensor] = field(default=None, init=True)
    indexes: Optional[List[int]] = field(default=None, init=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done,
            'weights': self.weights,
            'indexes': self.indexes,
        }


@dataclass
class TransitionEntry(BaseEntry):
    state: Tensor
    action: Tensor
    reward: float
    next_state: Tensor
    done: bool
    value: Optional[Tensor] = None,
    log_probs: Optional[Tensor] = None,
    action_mask: Optional[Tuple[np.ndarray]] = None,

    def to_dict(self) -> Dict[str, Any]:
        return {
            'state': self.state,
            'action': self.action,
            'reward': self.reward,
            'next_state': self.next_state,
            'done': self.done,
            'value': self.value,
            'log_probs': self.log_probs,
            'action_mask': self.action_mask,
        }


@dataclass
class RolloutEntry(BaseEntry):
    states: Tensor
    actions: Tensor
    values: Tensor
    log_probs: Tensor
    advantages: Tensor
    returns: Tensor
    action_masks: List[Tuple[np.ndarray, ...]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'states': self.states,
            'actions': self.actions,
            'values': self.values,
            'log_probs': self.log_probs,
            'advantages': self.advantages,
            'returns': self.returns,
            'action_masks': self.action_masks
        }


@dataclass
class RawTransition(BaseEntry):
    zone_state_wrapper: StateWrapper
    state_wrapper: StateWrapper
    zone_action: ZoneAction
    action: RepositionAction
    reward: float
    zone_next_state_wrapper: Optional[StateWrapper]
    next_state_wrapper: Optional[StateWrapper]
    done: bool
    step_info: Dict[str, Any]
    zone_value: Optional[Tensor] = None,
    zone_log_probs: Optional[Tensor] = None
    zone_action_mask: Optional[Tuple[np.ndarray, ...]] = None
    value: Optional[Tensor] = None
    log_probs: Optional[Tensor] = None
    action_mask: Optional[Tuple[np.ndarray, ...]] = None
    policy_index: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'zone_state_wrapper': self.zone_state_wrapper,
            'state_wrapper': self.state_wrapper,
            'zone_action': self.zone_action,
            'action': self.action,
            'reward': self.reward,
            'zone_next_state_wrapper': self.zone_next_state_wrapper,
            'next_state_wrapper': self.next_state_wrapper,
            'done': self.done,
            'step_info': self.step_info,
            'zone_value': self.zone_value,
            'zone_log_probs': self.zone_log_probs,
            'zone_action_mask': self.zone_action_mask,
            'value': self.value,
            'log_probs': self.log_probs,
            'action_mask': self.action_mask,
            'policy_index': self.policy_index,
        }
