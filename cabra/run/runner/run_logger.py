from logging import Logger
from typing import Dict, Tuple

import numpy as np

from cabra import SingleRunConfig
from cabra.common.data_structure import RunMode
from cabra.common.stats_tracker import Tracker, last_item


class RunLogger:

    def __init__(
            self,
            config: SingleRunConfig,
            logger: Logger,
            tracker: Tracker,
            training_steps: int,
            level: int = 20
    ):
        self.config: SingleRunConfig = config
        self.logger: Logger = logger
        self.stats_tracker: Tracker = tracker
        self.training_steps: int = training_steps
        self.level: int = level
        self.n_eval_seeds: int = len(self.config.random_seeds.evaluation)
        self.info_frequency = self.config.run.info_frequency
        self.debug_frequency = self.config.run.debug_frequency
        self.rollout_size: int = 0

    def _can_log(self, step: int) -> Tuple[bool, int]:
        can_log = False
        level = self.level
        if step % self.info_frequency == 0:
            can_log = True
            level = 20
        elif step % self.debug_frequency == 0:
            can_log = True
            level = 10
        return can_log, level

    def log(self, message, *args, level=None, **kwargs):
        log_level = level if level is not None else self.level
        self.logger.log(log_level, message, *args, **kwargs)

    def log_initialized_system(self, run_seed, n_zones: int):
        self.log(f'Initialized environment and agent - main configs: seed: {run_seed} '
                 f'| data model: {self.config.emulator.model.type} '
                 f'| truck: {self.config.environment.trucks.n_trucks} '
                 f'| zones: {n_zones} '
                 f'| nodes: {self.config.environment.nodes.n_nodes}')

    def log_initialized_agent(self, device, bootstrapping_steps: int):
        message = f'Initialized zone agent {self.config.environment.zones.agent.type}' \
                  f' and agent {self.config.environment.agent.type} using device {device}'
        if bootstrapping_steps > 0:
            message += f' and bootstrapping of {bootstrapping_steps} steps'
        self.log(message)

    def log_loaded_agent_model(self, model_path: str):
        self.log(f'Loaded agent state from path: {model_path}')

    def log_training_start(self, bootstrap_steps: int):
        message = f'Starting training run using {self.config.environment.agent.type} agent - '
        if bootstrap_steps > 0:
            message += f'Bootstrapping phase {bootstrap_steps} steps - '
        message += f'Training phase {self.training_steps} steps'
        self.log(message)

    def log_training_phase_start(self):
        self.log(f'Starting training phase for {self.training_steps} steps')

    def log_evaluation_start(self):
        self.log(f'Starting evaluation')

    def log_on_policy_train_step(self, iteration: int, eval_count: int, learning_params: Dict[str, str]):
        can_log, level = self._can_log(iteration)
        if can_log:
            message = self.get_train_step_message(iteration, eval_count, learning_params, is_on_policy=True)
            self.log(message, level=level)

    def log_off_policy_train_step(self, iteration: int, eval_count: int, learning_params: Dict[str, str]):
        can_log, level = self._can_log(iteration)
        if can_log:
            message = self.get_train_step_message(iteration, eval_count, learning_params, is_on_policy=False)
            self.log(message, level=level)

    def log_evaluation_run_completed(self, eval_count: int, run_mode: RunMode):
        eval_data = self.get_evaluation_data(aggr_fn=last_item)
        message = f'Phase: {run_mode.value} - Completed evaluation number {eval_count} - ' \
                  f'Average evaluation performance over {self.n_eval_seeds} random seeds: '
        for key, value in eval_data.items():
            message += f'{key}: {value} | '
        message = message[:-3]
        self.log(message)

    def log_evaluation_run_async(self, eval_count: int, run_mode: RunMode, eval_data):
        message = f'Phase: {run_mode.value} - Completed evaluation number {eval_count} - ' \
                  f'Average evaluation performance over {self.n_eval_seeds} random seeds: '
        for key, value in eval_data.items():
            message += f'{key}: {value} | '
        message = message[:-3]
        self.log(message)

    def log_new_best_evaluation_score(self, eval_count: int, run_mode: RunMode):
        best_eval_data = self.get_best_run_evaluation_data()
        message = f'Phase: {run_mode.value} - New best score for evaluation number {eval_count} - ' \
                  f'Best average evaluation performance over {self.n_eval_seeds} random seeds: '
        for key, value in best_eval_data.items():
            message += f'{key}: {value} | '
        message = message[:-3]
        self.log(message)

    def log_new_best_evaluation_score_async(self, eval_count: int, run_mode: RunMode, best_eval_data):
        message = f'Phase: {run_mode.value} - New best score for evaluation number {eval_count} - ' \
                  f'Best average evaluation performance over {self.n_eval_seeds} random seeds: '
        for key, value in best_eval_data.items():
            message += f'{key}: {value} | '
        message = message[:-3]
        self.log(message)

    def log_bootstrapping_start(self, bootstrap_steps: int):
        self.log(f'Starting bootstrapping phase for {bootstrap_steps} steps')

    def log_bootstrapping_status(self, steps: int, bootstrap_steps: int):
        can_log, level = self._can_log(steps)
        if can_log:
            self.log(f'Phase: bootstrapping | steps {steps}/{bootstrap_steps}', level=level)

    def log_bootstrapping_end(self, bootstrap_steps: int):
        self.log(f'Completed bootstrapping phase after {bootstrap_steps} steps')

    def log_on_policy_training_completed(self, best_eval_iteration: int):
        message = self.get_training_completed_message(best_eval_iteration)
        if message is not None:
            self.log(message)

    def log_off_policy_training_completed(self, best_eval_iteration: int):
        message = self.get_training_completed_message(best_eval_iteration)
        if message is not None:
            self.log(message)

    def log_evaluation_start_info(self, eval_count: int):
        self.log(f'Starting evaluation iteration {eval_count} using {self.n_eval_seeds} random seeds ')

    def get_training_completed_message(self, best_eval_iteration: int) -> str:
        message = f'Completed training after {self.training_steps} learn steps - ' \
                  f'Best average evaluation performance over {self.n_eval_seeds} random seeds, ' \
                  f'obtained at evaluation {best_eval_iteration}: '
        best_eval_data = self.get_best_run_evaluation_data()
        for key, value in best_eval_data.items():
            message += f'{key}: {value} | '
        message = message[:-3]
        return message

    def get_evaluation_data(self, aggr_fn: callable = np.max) -> Dict[str, str]:
        prefix = RunMode.Eval.value
        total_steps = self.stats_tracker.get_key_to_string(f'{prefix}/avg/total_steps', aggr_fn=aggr_fn)
        solved_steps = self.stats_tracker.get_key_to_string(f'{prefix}/avg/solved_steps', aggr_fn=aggr_fn)
        ignore = ['total_steps', 'solved_steps']
        data = {
            'avg solved steps': f'{solved_steps}/{total_steps}',
        }
        for metric in self.stats_tracker.rollout_variables_eval:
            if metric not in ignore:
                value = self.stats_tracker.get_key_to_string(f'{prefix}/avg/{metric}', aggr_fn=aggr_fn)
                data[f'avg {metric.replace("_", " ")}'] = value

        return data

    def get_best_run_evaluation_data(self) -> Dict[str, str]:
        best_run_prefix = f'{RunMode.Eval}/best_run'
        best_run_variables = self.stats_tracker.rollout_variables_eval
        solved_steps = self.stats_tracker.get_key_to_string(f'{best_run_prefix}/solved_steps')
        total_steps = self.stats_tracker.get_key_to_string(f'{best_run_prefix}/total_steps')
        ignore = ['total_steps', 'solved_steps']
        data = {
            'avg solved steps': f'{solved_steps}/{total_steps}',
        }
        for var in best_run_variables:
            if var not in ignore:
                data[f'avg {var.replace("_", " ")}'] = self.stats_tracker.get_key_to_string(f'{best_run_prefix}/{var}')

        return data

    def get_training_data(self, iteration: int, is_on_policy: bool = False) -> Dict[str, str]:
        prefix = RunMode.Train.value
        metrics_window = self.config.run.metrics_window
        step_window = 1 if is_on_policy else metrics_window
        total_steps = self.stats_tracker.get_key_to_string(f'{prefix}/avg-{step_window}-steps/total_steps',
                                                           aggr_fn=last_item)
        train_steps = self.stats_tracker.get_key_to_string(f'{prefix}/training_steps')
        episodes = self.stats_tracker.get_key_to_string(f'{prefix}/episodes')

        data = {
            'total steps': total_steps,
            'episodes': episodes,
        }
        if is_on_policy:
            data['rollout steps'] = self.rollout_size
            data['learning steps'] = train_steps

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

    def get_train_step_message(
            self,
            iteration: int,
            eval_count: int,
            learning_params: Dict[str, str],
            is_on_policy: bool
    ) -> str:
        message = f'Phase: training - train steps: {iteration}/{self.training_steps} - train metrics: '
        training_data = self.get_training_data(iteration, is_on_policy)
        for key, value in training_data.items():
            message += f'{key}: {value} | '
        message = message[:-3]
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
