from typing import List, Tuple

import numpy as np

from cabra.environment.data_structure import StateType
from cabra.environment.state_wrapper import StateWrapper


class Rollout:

    def __init__(
            self,
            obs: List[StateWrapper],
            acs: List[np.ndarray],
            rewards: List[np.ndarray],
            next_obs: List[StateWrapper],
    ):
        self.states: List[StateWrapper] = obs
        self.rewards: np.ndarray = np.array(rewards, dtype=np.float32)
        self.actions: np.ndarray = np.array(acs, dtype=np.float32)
        self.next_states: List[StateWrapper] = next_obs

    def get_states(self, state_type: StateType) -> np.ndarray:
        return np.array([state_wrapper[state_type] for state_wrapper in self.states], dtype=np.float32)

    def get_actions(self) -> np.ndarray:
        return self.actions

    def get_rewards(self) -> np.ndarray:
        return self.rewards

    def get_next_states(self, state_type: StateType) -> np.ndarray:
        return np.array([state_wrapper[state_type] for state_wrapper in self.next_states], dtype=np.float32)

    def to_dict(self):
        return {
            'states': self.states,
            'rewards': self.rewards,
            'actions': self.actions,
            'next_states': self.next_states,
        }

    def to_tuple(
            self,
            state_type: StateType
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return \
            np.array([state_wrapper[state_type] for state_wrapper in self.states], dtype=np.float32), \
                self.actions, \
                np.array([state_wrapper[state_type] for state_wrapper in self.next_states], dtype=np.float32), \
                self.rewards

    def __str__(self):
        return f'<Rollout size={len(self)}>'

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.rewards)

    # def __getattr__(self, item: str):
    #     attr = getattr(self, item, None)
    #     if attr is not None:
    #         return attr


def convert_list_of_rollouts(
        paths: List[Rollout]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    states = np.concatenate([path.states for path in paths])
    actions = np.concatenate([path.actions for path in paths])
    next_states = np.concatenate([path.next_states for path in paths])
    concatenated_rewards = np.concatenate([path.rewards for path in paths])
    return states, actions, next_states, concatenated_rewards
