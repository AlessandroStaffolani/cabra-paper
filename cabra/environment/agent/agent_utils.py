from cabra.environment.agent import AgentType
from cabra.environment.agent.baseline import RandomAgent, DoNothingAgent, ConstrainedRandomAgent, \
    ConstrainedGreedyAgent, NStepsOracleAgent
from cabra.environment.agent.baseline.heuristics import (
    SimulatedAnnealingAgent, ConstrainedSimulatedAnnealingAgent)
from cabra.environment.agent.ppo import PPOAgent, ConstrainedPPOAgent

AGENTS_MAPPING = {
    AgentType.Random: RandomAgent,
    AgentType.DoNothing: DoNothingAgent,
    AgentType.PPO: PPOAgent,
    AgentType.ConstrainedPPO: ConstrainedPPOAgent,
    AgentType.ConstrainedRandom: ConstrainedRandomAgent,
    AgentType.ConstrainedGreedy: ConstrainedGreedyAgent,
    AgentType.NStepsOracle: NStepsOracleAgent,
    AgentType.SimulatedAnnealing: SimulatedAnnealingAgent,
    AgentType.ConstrainedSimulatedAnnealing: ConstrainedSimulatedAnnealingAgent
}
