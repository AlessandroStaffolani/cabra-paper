from abc import abstractmethod
from typing import List, Dict, Union, Any

from cabra import SingleRunConfig
from cabra.common.data_structure import RunMode
from cabra.core.step import Step
from cabra.environment.data_structure import RewardFunctionType
from cabra.environment.node import Node, DistancesProvider
from cabra.environment.state_wrapper import StateWrapper


class RewardFunctionTypeError(Exception):

    def __init__(self, name: RewardFunctionType, *args):
        message = f'No reward class mapping for type: {name}'
        super(RewardFunctionTypeError, self).__init__(message, *args)


class RewardAbstract:

    def __init__(
            self,
            name: str,
            config: SingleRunConfig,
            run_mode: RunMode,
            distances_provider: DistancesProvider,
            disable_cost: bool = False,
            normalize_cost: bool = True,
            **kwargs):
        self.name: str = name
        self.config: SingleRunConfig = config
        self.env_config = self.config.environment
        self.disable_cost: bool = disable_cost
        self.normalize_cost: bool = normalize_cost
        self.distances_provider: DistancesProvider = distances_provider
        self.run_mode: RunMode = run_mode
        self.training_scaler: float = self.config.environment.reward.training_scaler
        self.solved_bonus: float = self.config.environment.reward.solved_bonus

    def __str__(self):
        return f'<Reward name={self.name} >'

    def reward_info(self) -> Dict[str, Any]:
        return {}

    @property
    def reward_scaler(self) -> float:
        if self.run_mode == RunMode.Train:
            return self.training_scaler
        else:
            return 1

    @property
    def solved_reward_bonus(self) -> float:
        if self.run_mode == RunMode.Train:
            return self.solved_bonus
        else:
            return 0

    @abstractmethod
    def compute(
            self,
            state_wrapper: StateWrapper,
            action_cost: float,
            step: Step,
            nodes: List[Node],
            shortages: int,
            env_shortages: int,
            penalty: Union[int, float] = 0,
            **kwargs
    ) -> float:
        pass

    def reset(self):
        pass


class GlobalShortageAndCostReward(RewardAbstract):

    def __init__(
            self,
            config: SingleRunConfig,
            run_mode: RunMode,
            distances_provider: DistancesProvider,
            disable_cost: bool = False,
            shortage_weight: float = 0.3,
            environment_shortage_weight: float = 0.6,
            cost_weight: float = 0.1,
            **kwargs
    ):
        super(GlobalShortageAndCostReward, self).__init__(
            name=RewardFunctionType.GlobalShortageAndCost,
            config=config,
            run_mode=run_mode,
            distances_provider=distances_provider,
            disable_cost=disable_cost,
            **kwargs
        )
        self.shortage_weight: float = shortage_weight
        self.environment_shortage_weight: float = environment_shortage_weight
        self.cost_weight: float = cost_weight
        self.current_shortages: int = 0
        self.current_env_shortages: int = 0
        self.current_action_cost: float = 0
        self.current_bonus: float = 0
        self.current_penalty: float = 0

        # assert self.cost_weight + self.shortage_weight + self.environment_shortage_weight == 1, \
        #     'shortages, env_shortages and cost weights must sum to 1'

    def compute(
            self,
            state_wrapper: StateWrapper,
            action_cost: float,
            step: Step,
            nodes: List[Node],
            shortages: int,
            env_shortages: int,
            penalty: Union[int, float] = 0,
            **kwargs
    ) -> float:
        if self.disable_cost:
            action_cost = 0
        env_shortages_rew = -(self.environment_shortage_weight * env_shortages * self.reward_scaler)
        shortages_rew = -(shortages * self.shortage_weight * self.reward_scaler)
        cost_rew = -(action_cost * self.cost_weight * self.reward_scaler)
        bonus = 0
        if env_shortages == 0 and shortages == 0:
            # problem solved, we can provide the bonus
            bonus = self.solved_reward_bonus * self.reward_scaler
        reward = env_shortages_rew + shortages_rew + cost_rew + bonus + penalty
        self.current_shortages = shortages
        self.current_env_shortages = env_shortages
        self.current_action_cost = action_cost
        self.current_bonus = bonus
        self.current_penalty = penalty
        return reward

    def reward_info(self) -> Dict[str, Any]:
        return {
            'cost': self.current_action_cost,
            'shortages': self.current_shortages,
            'env_shortages': self.current_env_shortages,
            'bonus': self.current_bonus,
            'penalty': self.current_penalty,
        }

    def reset(self):
        self.current_shortages: int = 0
        self.current_env_shortages: int = 0
        self.current_action_cost: float = 0
        self.current_bonus: float = 0
        self.current_penalty: float = 0


REWARD_CLASS_MAPPING = {
    RewardFunctionType.GlobalShortageAndCost: GlobalShortageAndCostReward,
}


def get_reward_class(
        reward_type: RewardFunctionType,
        config,
        run_mode: RunMode,
        distances_provider: DistancesProvider,
        disable_cost: bool = False,
        **additional_parameters
):
    if reward_type in REWARD_CLASS_MAPPING:
        rew_class = REWARD_CLASS_MAPPING[reward_type]
        return rew_class(
            config=config,
            run_mode=run_mode,
            disable_cost=disable_cost,
            distances_provider=distances_provider,
            **additional_parameters)
    else:
        raise RewardFunctionTypeError(reward_type)
