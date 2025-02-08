from cabra.common.enum_utils import ExtendedEnum


class AgentType(str, ExtendedEnum):
    Random = 'random'
    DoNothing = 'do-nothing'
    PPO = 'ppo-fs'
    ConstrainedPPO = 'cabra'
    ConstrainedRandom = 'constrained-random'
    ConstrainedGreedy = 'constrained-greedy'
    NStepsOracle = 'n-steps-oracle'
    SimulatedAnnealing = 'simulated-annealing'
    ConstrainedSimulatedAnnealing = 'constrained-simulated-annealing'

    def is_on_policy(self) -> bool:
        ac_types = [AgentType.PPO, AgentType.ConstrainedPPO]
        return self in ac_types

    def use_validation(self) -> bool:
        validation_types = [AgentType.PPO, AgentType.ConstrainedPPO]
        return self in validation_types

    def has_loss(self) -> bool:
        loss_types = [AgentType.PPO, AgentType.ConstrainedPPO]
        return self in loss_types

    def is_off_policy(self) -> bool:
        v_based_types = []
        return self in v_based_types

    def is_baseline(self) -> bool:
        baselines = [AgentType.Random, AgentType.ConstrainedRandom, AgentType.DoNothing, AgentType.ConstrainedGreedy,
                     AgentType.NStepsOracle, AgentType.SimulatedAnnealing, AgentType.ConstrainedSimulatedAnnealing]
        return self in baselines

    def is_oracle(self) -> bool:
        oracle_baselines = [AgentType.NStepsOracle]
        return self in oracle_baselines

    def is_ppo(self):
        ppo_agent_types = [AgentType.PPO, AgentType.ConstrainedPPO]
        return self in ppo_agent_types

    def is_meta_heuristic(self) -> bool:
        meta_heuristics = [AgentType.SimulatedAnnealing, AgentType.ConstrainedSimulatedAnnealing]
        return self in meta_heuristics
