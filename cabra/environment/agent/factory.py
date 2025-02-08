from logging import Logger
from typing import Optional, Dict, List

from numpy.random import RandomState

from cabra import SingleRunConfig
from cabra.common.data_structure import RunMode
from cabra.environment.action_space import AbstractActionSpace
from cabra.environment.agent import AgentType
from cabra.environment.agent.abstract import AgentAbstract
from cabra.environment.agent.agent_utils import AGENTS_MAPPING
from cabra.environment.node import Node, DistancesProvider
from cabra.environment.state_builder import StateBuilder
from cabra.environment.data_structure import StateType, ActionType
from cabra.environment.zone import Zone
from cabra.environment.zones_env_wrapper import ZonesEnvWrapper


def create_agent(
        agent_type: AgentType,
        action_space: AbstractActionSpace,
        random_state: RandomState,
        action_spaces: Dict[ActionType, int],
        state_spaces: Dict[StateType, int],
        nodes: List[Node],
        zones: Dict[str, Zone],
        nodes_max_distance: float,
        log: Logger,
        state_builder: StateBuilder,
        is_zone_agent: bool,
        config: Optional[SingleRunConfig] = None,
        mode: RunMode = RunMode.Train,
        distances_provider: Optional[DistancesProvider] = None,
        env: Optional[ZonesEnvWrapper] = None,
        **parameters
) -> AgentAbstract:
    if agent_type in AGENTS_MAPPING:
        return AGENTS_MAPPING[agent_type](
            action_space=action_space,
            random_state=random_state,
            action_spaces=action_spaces,
            state_spaces=state_spaces,
            nodes=nodes,
            zones=zones,
            nodes_max_distance=nodes_max_distance,
            log=log,
            state_builder=state_builder,
            is_zone_agent=is_zone_agent,
            config=config,
            mode=mode,
            distances_provider=distances_provider,
            env=env,
            **parameters
        )
    else:
        raise AttributeError(f'AgentType "{agent_type}" not available')
