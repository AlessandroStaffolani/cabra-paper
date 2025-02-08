from typing import Optional

from numpy.random import RandomState

from cabra.common.data_structure import RunMode
from cabra.common.mpi.mpi_tools import proc_id, num_procs, mpi_sum
from cabra.common.stats_tracker import last_item
from cabra.run import RunStatus
from cabra.run.runner import RemoteRunner
from cabra.run.runner.mpi_run_logger import MPIRunLogger
from cabra.run.runner.utils import clean_log_folder


class MPIRunner(RemoteRunner):

    def __init__(
            self,
            run_code: str,
            test_run: bool = False,
    ):
        self.rank: int = proc_id()
        self.processes: int = num_procs()
        self.is_root_process: bool = self.rank == 0
        self.run_code = run_code
        self.test_run: bool = test_run
        super(MPIRunner, self).__init__(run_code=self.run_code)
        self.global_training_steps: int = self.training_steps * self.processes
        # override the run_logger with the mpi_run_logger
        self.run_logger: MPIRunLogger = MPIRunLogger(
            self.config, self.logger, self.stats_tracker, self.training_steps, self.global_training_steps)

    def _init_seeds(self):
        if self.run_mode == RunMode.Eval:
            run_seed = self.config.random_seeds.evaluation[0]
        else:
            run_seed = self.config.random_seeds.training
        self.random_seed = run_seed + (10000 * (self.rank + 1))
        self.run_random: RandomState = RandomState(self.random_seed)

    def _init_eval_pool(self):
        if self.is_root_process:
            super()._init_eval_pool()

    def _init_redis(self):
        if self.is_root_process:
            super()._init_redis()

    def _update_redis_stats(self, current_step: int):
        if self.run_mode == RunMode.Train:
            prefix = RunMode.Train.value
            metrics_window = 1 if self.is_on_policy else self.config.run.metrics_window
            global_total_steps = mpi_sum(
                self.stats_tracker.get_key_to_string(f'{prefix}/avg-{metrics_window}-steps/total_steps',
                                                     aggr_fn=last_item))
            global_train_steps = mpi_sum(self.stats_tracker.get_key_to_string(f'{prefix}/training_steps'))
            global_episodes = mpi_sum(self.stats_tracker.get_key_to_string(f'{prefix}/episodes'))
            global_current_steps = mpi_sum(current_step)
            global_total_iterations = mpi_sum(self.training_steps)
        if self.is_root_process:
            if self.redis is not None:
                run_stats = {}
                if self.run_mode == RunMode.Train:
                    run_stats = {
                        'global_iterations': global_current_steps,
                        'global_total iterations': global_total_iterations,
                        'global_steps': global_total_steps,
                        'global_episodes': global_episodes,
                        'global_learning steps': global_train_steps,
                    }
                run_stats = {
                    **run_stats,
                    **self.stats_tracker.redis_stats(get_max_and_last=False),
                    'evaluation_counter': self.completed_eval_counter
                }
                stats = {}
                for key, value in run_stats.items():
                    stats[key.replace('avg-1-steps', 'last_rollout')] = value
                if self.best_eval_performance is not None:
                    stats['best_evaluation_performance'] = self.best_eval_performance
                    stats['best_evaluation_performance_iteration'] = self.best_eval_performance_iteration
                self.redis.add_stats(
                    current_step=current_step,
                    status=RunStatus.RUNNING,
                    run_worker=self.run_worker,
                    run_stats=stats,
                    agent_name=self.agent.name,
                    zone_agent_name=self.zone_agent.name,
                    multi_run_code=self.config.multi_run.multi_run_code,
                    multi_run_params=self.config.multi_run.multi_run_params
                )

    def save_agent_model(self, agent_model, filename: str):
        if self.is_root_process and not self.test_run:
            super().save_agent_model(agent_model, filename)

    def _save_evaluation_data(self):
        if self.is_root_process and not self.test_run:
            super()._save_evaluation_data()

    def _after_run(self, run_performance: Optional[float]):
        if self.is_root_process and not self.test_run:
            super()._after_run(run_performance)

    def close_runner(self):
        self.run_stats_manager.close()
        if self.is_root_process:
            if self.run_mode == RunMode.Train:
                if self.agent.name.is_on_policy():
                    # log final info
                    self.run_logger.log_on_policy_training_completed(self.best_eval_performance_iteration + 1)
                elif self.agent.name.is_off_policy():
                    # log final info
                    self.run_logger.log_off_policy_training_completed(self.best_eval_performance_iteration + 1)
            if self.redis is not None:
                self.redis.close()
        self.mongo.close()
        if self.is_root_process:
            clean_log_folder(run_code=self.run_code, run_config=self.config)
            del self.logger

    def do_evaluations(self, agent_state_model, zone_agent_state_model):
        if self.is_root_process:
            super().do_evaluations(agent_state_model, zone_agent_state_model)
