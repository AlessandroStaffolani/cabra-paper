from typing import Tuple

import numpy as np

from cabra.common.enum_utils import ExtendedEnum
from cabra.common.math_util import normalize_scalar


class ScalarizationFunction(str, ExtendedEnum):
    Linear = 'linear'
    LinearCostWeighted = 'linear-cost-weighted'
    SquaredGapCostWeighted = 'squared-gap-cost-weighted'
    SquaredRemainingGap = 'squared-remaining-gap'
    LogSatisfiedMultiplier = 'log-satisfied-multiplier'


def linear_scalarization(remaining_gap: float, cost: float,
                         initial_gap: float, max_cost: float, alpha: float,
                         clipped_remaining_gap: bool = False, normalization_range: Tuple[int, int] = (0, 1),
                         normalize_objectives: bool = True,
                         **kwargs
                         ) -> float:
    if normalize_objectives:
        # normalize the two rewards between 0 and 10
        if clipped_remaining_gap:
            if remaining_gap == 0:
                normalized_gap = 1
            else:
                normalized_gap = 0
        else:
            normalized_gap = normalize_scalar(remaining_gap, 0, initial_gap,
                                              normalization_range[0], normalization_range[1])
        action_cost = -cost
        normalized_action_cost = normalize_scalar(action_cost, 0, -max_cost,
                                                  normalization_range[0], normalization_range[1])
    else:
        normalized_gap = remaining_gap
        normalized_action_cost = -cost
    return alpha * normalized_gap + (1 - alpha) * normalized_action_cost


def linear_cost_weighted(remaining_gap: float, cost: float, alpha: float, **kwargs) -> float:
    return remaining_gap - alpha*cost


def squared_gap_cost_weighted(remaining_gap: float, cost: float, alpha: float, **kwargs) -> float:
    squared = remaining_gap ** 2
    return -squared - alpha*cost


def squared_remaining_gap(remaining_gap: float, cost: float,
                          alpha: float, **kwargs) -> float:
    squared = remaining_gap ** 2
    return alpha * -squared + (1-alpha) * cost


def log_satisfied_multiplier(remaining_gap: float, cost: float,
                             alpha: float, satisfied_nodes: int, n_nodes: int,
                             **kwargs) -> float:
    multiplier = np.log(n_nodes/satisfied_nodes)
    return multiplier * remaining_gap + alpha * cost


MAPPING = {
    ScalarizationFunction.Linear: linear_scalarization,
    ScalarizationFunction.LinearCostWeighted: linear_cost_weighted,
    ScalarizationFunction.SquaredGapCostWeighted: squared_gap_cost_weighted,
    ScalarizationFunction.SquaredRemainingGap: squared_remaining_gap,
    ScalarizationFunction.LogSatisfiedMultiplier: log_satisfied_multiplier
}


def scalarize_rewards(func_type: ScalarizationFunction, remaining_gap: float, cost: float, **scal_func_params) -> float:
    if func_type in MAPPING:
        return MAPPING[func_type](remaining_gap, cost, **scal_func_params)
    else:
        raise AttributeError(f'Scalarization Function {func_type} has no scalarization function mapping')
