from logging import Logger
from typing import Dict

import numpy as np

from cabra import SingleRunConfig
from cabra.common.data_structure import RunMode
from cabra.common.mpi.mpi_tools import proc_id, num_procs, mpi_sum
from cabra.common.stats_tracker import Tracker, last_item
from cabra.run.runner.run_logger import RunLogger


class MPIRunLogger(RunLogger):

    def __init__(
            self,
            config: SingleRunConfig,
            logger: Logger,
            tracker: Tracker,
            training_steps: int,
            global_training_steps: int,
            level: int = 20
    ):
        self.rank: int = proc_id()
        self.processes: int = num_procs()
        self.is_root_process: bool = self.rank == 0
        super().__init__(config, logger, tracker, training_steps, level)
        self.global_training_steps: int = global_training_steps
        self.train_iteration_counter: int = 0

    def log(self, message, *args, level=None, **kwargs):
        message = f'Worker {self.rank + 1}/{self.processes} - {message}'
        super().log(message, *args, level=level, **kwargs)

    def log_training_start(self, bootstrap_steps: int):
        message = f'Starting training run using {self.config.environment.agent.type} agent - '
        if bootstrap_steps > 0:
            message += f'Bootstrapping phase {bootstrap_steps} steps - '
        message += f'Training phase {self.global_training_steps} steps - ' \
                   f'training steps per worker {self.training_steps}'
        self.log(message)

    def get_training_completed_message(self, best_eval_iteration: int) -> str:
        if self.is_root_process:
            return super().get_training_completed_message(best_eval_iteration)

    def get_train_step_message(
            self,
            iteration: int,
            eval_count: int,
            learning_params: Dict[str, str],
            is_on_policy: bool
    ) -> str:
        self.train_iteration_counter = iteration
        global_iteration = mpi_sum(self.train_iteration_counter)
        message = f'Phase: training - global train steps: {global_iteration}/{self.global_training_steps} ' \
                  f'- worker train steps: {iteration}/{self.training_steps} - train metrics: '
        training_data = self.get_training_data(iteration, is_on_policy)
        for key, value in training_data.items():
            message += f'{key}: {value} | '
        message = message[:-3]
        if self.is_root_process:
            if eval_count > 0:
                message += ' - best eval performance: '
                best_eval_data = self.get_best_run_evaluation_data()
                for key, value in best_eval_data.items():
                    message += f'{key}: {value} | '
                message = message[:-3]
        message += ' - learning metrics: '
        for key, value in learning_params.items():
            message += f'{key}: {value} | '
        message = message[:-3]
        return message

    def get_training_data(self, iteration: int, is_on_policy: bool = False) -> Dict[str, str]:
        prefix = RunMode.Train.value
        metrics_window = self.config.run.metrics_window
        step_window = 1 if is_on_policy else metrics_window
        total_steps = self.stats_tracker.get_key_to_string(f'{prefix}/avg-{step_window}-steps/total_steps',
                                                           aggr_fn=last_item)
        train_steps = self.stats_tracker.get_key_to_string(f'{prefix}/training_steps')
        episodes = self.stats_tracker.get_key_to_string(f'{prefix}/episodes')

        data = {
            'total steps': mpi_sum(total_steps),
            'total episodes': mpi_sum(episodes),
            'total learning steps': mpi_sum(train_steps),
            'rollout steps': self.rollout_size,
            'worker steps': total_steps,
            'worker episodes': episodes,
            'worker learning steps': train_steps,
        }

        if is_on_policy:
            window_check = iteration
        else:
            window_check = int(total_steps)

        if window_check > metrics_window or is_on_policy:
            field_prefix = 'last rollout' if is_on_policy else f'last {metrics_window}'
            field_suffix = '' if is_on_policy else ' avg'

            for metric_name in self.stats_tracker.rollout_variables:
                if metric_name != 'total_steps':
                    key = f'{field_prefix}{field_suffix} {metric_name.replace("_", " ")}'
                    stats_key = f'{prefix}/avg-{step_window}-steps/{metric_name}'
                    data[key] = self.stats_tracker.get_key_to_string(stats_key, aggr_fn=np.mean)

        if self.stats_tracker.get(f'{prefix}/episode/{self.stats_tracker.episode_variables[0]}') is not None:
            for metric_name in self.stats_tracker.episode_variables:
                key = f'last {metrics_window} episodes {metric_name.replace("_", " ")}'
                data[key] = self.stats_tracker.get_key_to_string(f'{prefix}/episode/{metric_name}')

        return data
