from typing import Union

import numpy as np
import torch

from cabra.common.enum_utils import ExtendedEnum
from cabra.environment.action_space import SubActionSpace


class ExplorationPolicy(str, ExtendedEnum):
    Random = 'random'
    Greedy = 'epsilon'


def random_policy(action_space: SubActionSpace, random_state: np.random.RandomState) -> int:
    return int(action_space.sample(random_state=random_state))


def greedy_policy(
        q_values: Union[torch.Tensor, np.ndarray],
        action_space: SubActionSpace,
        device=torch.device('cpu')
) -> int:
    if isinstance(q_values, torch.Tensor):
        min_value = torch.tensor(float(np.finfo(np.float).min), device=device)
        disabled_actions = action_space.get_disabled_actions()
        disabled_actions = torch.tensor(
            disabled_actions.tolist() + np.arange(action_space.size(), q_values.shape[-1]).tolist(),
            dtype=torch.long, device=device)
        q_values.index_fill_(1, disabled_actions, min_value.item())
        return int(q_values.max(1)[1].item())
    else:
        min_value = float(np.finfo(np.float32).min)
        disabled_actions = action_space.get_disabled_actions()
        q_values[disabled_actions] = min_value
        return int(q_values.argmax())
