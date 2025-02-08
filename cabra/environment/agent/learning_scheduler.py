from typing import Union, Callable

from cabra.common.enum_utils import ExtendedEnum

Schedule = Callable[[float], float]


class LRScheduler(str, ExtendedEnum):
    Constant = 'constant'
    Linear = 'linear'


def linear_schedule(initial_value: Union[float, str]) -> Schedule:
    """
    Linear learning rate schedule.
    """
    # Force conversion to float
    initial_value_ = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value_

    return func


def constant_schedule(val: float) -> Schedule:
    """
    Create a function that returns a constant
    It is useful for learning rate schedule (to avoid code duplication)
    """

    def func(_):
        return val

    return func


MAPPING = {
    LRScheduler.Constant: constant_schedule,
    LRScheduler.Linear: linear_schedule
}


def get_scheduler(lr_scheduler: LRScheduler, initial_value: Union[float, int]) -> Schedule:
    if lr_scheduler in MAPPING:
        return MAPPING[lr_scheduler](initial_value)
    else:
        raise AttributeError('Invalid LR scheduler')
