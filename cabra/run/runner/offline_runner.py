import json
import os
from logging import Logger
from typing import Optional, Dict

import pandas as pd

from cabra import SingleRunConfig
from cabra.common.data_structure import RunMode
from cabra.common.filesystem import create_directory_from_filepath
from cabra.environment.agent import AgentType
from cabra.run.remote import PersistenceInfo
from cabra.run.run_stats_manager import RunStatsManager
from cabra.run.runner.base_runner import BaseRunner
from cabra.run.runner.utils import get_run_logger, clean_log_folder


class OfflineRunner(BaseRunner):

    def __init__(
            self,
            run_code: str,
            config: SingleRunConfig,
            run_worker: str,
    ):
        self.run_code: str = run_code
        self.run_worker: str = run_worker
        self.config: SingleRunConfig = config
        self.logger: Logger = get_run_logger(self.run_code, self.config)
        super().__init__(run_code=self.run_code, config=self.config, log=self.logger)
        self.run_stats_manager: Optional[RunStatsManager] = None
        self.result_path: Optional[PersistenceInfo] = None
        self.run_status_path: str = os.path.join(os.getenv('DATA_BASE_DIR'), 'workers', 'run_status')
        self.multi_run_code = self.config.multi_run.multi_run_code
        self.multi_run_params = self.config.multi_run.multi_run_params
        self._init()

    @property
    def run_stats_file(self):
        return os.path.join(self.run_status_path, f'{self.run_code}_stats.json')

    def _init(self):
        try:
            if not self.initialized:
                self._init_seeds()
                self._init_env()
                self._init_agent()
                self._init_trucks_wrapper()
                self._init_eval_pool()
                self._init_run_stats_manager()
                self._init_run_status_folder()
                self.initialized = True
                self.run_logger.log_initialized_system(self.random_seed, self.env.n_zones)
        except Exception as e:
            self.logger.exception(e)
            raise e

    def _init_run_stats_manager(self):
        self.run_stats_manager: RunStatsManager = RunStatsManager(
            run_code=self.run_code,
            config=self.config,
            logger=self.logger,
            nodes=self.env.nodes,
            agent_name=self.agent.name,
            total_steps=self.env.time_step.stop_step,
            step_size=self.env.time_step.step_size,
            object_handler=self.object_handler
        )
        self.stats_tracker = self.run_stats_manager.stats_tracker
        self.env.set_stats_tracker(self.run_stats_manager.stats_tracker)
        self.agent.set_stats_tracker(self.run_stats_manager.stats_tracker)
        self.zone_agent.set_stats_tracker(self.run_stats_manager.stats_tracker)
        self.run_logger.stats_tracker = self.stats_tracker
        self.result_path: PersistenceInfo = PersistenceInfo(
            path=self.run_stats_manager.save_folder,
            save_mode=self.config.saver.mode,
            host='offline'
        )

    def _get_base_stats(self, iteration: int = 0) -> dict:
        stats = {
            'run_code': self.run_code,
            'run_worker': self.run_worker,
            'current_iteration': iteration,
            'total_training_iterations': self.training_steps,
            'agent': self.agent.name.value,
            'zone_agent': self.zone_agent.name.value,
        }
        if self.multi_run_code is not None:
            stats['multi_run_code'] = self.multi_run_code
        if len(self.multi_run_params) > 0:
            for param_conf in self.multi_run_params:
                stats[param_conf['key']] = param_conf['value']
        return stats

    def _save_run_stats(self, stats: dict):
        with open(self.run_stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

    def _init_run_status_folder(self):
        self._save_run_stats(self._get_base_stats())

    def _update_run_stats(self, current_step: int):
        stats = self._get_base_stats(current_step)
        run_stats = self.stats_tracker.redis_stats(get_max_and_last=False)
        run_stats['evaluation_counter'] = self.completed_eval_counter
        if self.best_eval_performance is not None:
            run_stats['best_evaluation_performance'] = self.best_eval_performance
            run_stats['best_evaluation_performance_iteration'] = self.best_eval_performance_iteration
        for key, val in run_stats.items():
            stats[key] = val
        self._save_run_stats(stats)

    def after_train_step_callback(self, is_on_policy: bool, iteration: int):
        super().after_train_step_callback(is_on_policy, iteration)
        if iteration % self.config.run.info_frequency == 0:
            self._update_run_stats(iteration)

    def after_evaluation_run_callback(
            self,
            eval_agent,
            eval_performance: float,
            eval_counter: int,
            eval_zone_agent,
            eval_metrics: Dict[str, float],
            is_async: bool = False,
    ):
        super().after_evaluation_run_callback(
            eval_agent, eval_performance, eval_counter, eval_zone_agent, eval_metrics, is_async)
        df = pd.DataFrame(eval_metrics, index=[0])
        save_path = os.path.join(self.object_handler.base_path, self.result_path['path'], 'evaluations.csv')
        create_directory_from_filepath(save_path)
        df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)

    def save_agent_model(self, agent_model, filename: str):
        if self.run_mode == RunMode.Train:
            self.run_stats_manager.save_agent(
                agent_model, save_agent_state=self.agent.save_agent_state, filename=filename)
            self.logger.info(f'Save agent model {filename}')

    def after_run_callback(self):
        run_performance = super().after_run_callback()
        self._after_run(run_performance)

    def _after_run(self, run_performance: Optional[float]):
        self.run_stats_manager.save_stats(agent_state=None, save_agent_state=self.agent.save_agent_state)
        self._save_run_summary(run_performance)
        os.remove(self.run_stats_file)

    def _save_run_summary(self, run_performance: Optional[float]):
        if self.multi_run_code is not None:
            stats = {}
            for param in self.multi_run_params:
                stats[f'hp/{param.key_short}'] = param.value if param.value is not None else 'None'
            self._set_baselines_hp(stats)
            if run_performance is not None:
                stats['run_performance'] = run_performance
            db_stats = self.stats_tracker.db_stats()
            db_stats[f'{RunMode.Eval.value}/evaluations_counter'] = self.completed_eval_counter
            if self.best_eval_performance is not None:
                db_stats[f'{RunMode.Eval.value}/best/performance'] = self.best_eval_performance
                db_stats[
                    f'{RunMode.Eval.value}/best/performance_iteration'] = self.best_eval_performance_iteration
            for key, data in db_stats.items():
                if key.find(f'{RunMode.Train.value}/') == 0 or key.find(f'{RunMode.Eval.value}/') == 0 \
                        or key.find(f'{RunMode.Validation.value}/') == 0:
                    stats[key] = data
                elif key.find('metric/') == 0:
                    stats[key.replace('metric/', f'{RunMode.Train.value}/')] = data
            stats['multi_run_code'] = self.multi_run_code
            stats['agent'] = self.agent.name
            stats['zone_agent'] = self.zone_agent.name
            stats['run_code'] = self.run_code
            save_path = os.path.join(self.config.saver.get_base_path(), self.multi_run_code, 'summary.csv')
            df = pd.DataFrame(stats, index=[0])
            df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)

    def _set_baselines_hp(self, stats):
        if self.agent.name in [AgentType.ConstrainedGreedy, AgentType.ConstrainedRandom]:
            stats['hp/critical_threshold'] = self.config.environment.constrained_space.critical_threshold
            stats['hp/max_distance'] = self.config.environment.constrained_space.max_distance
            stats['hp/zone_max_distance'] = self.config.environment.constrained_space.zone_max_distance
            stats['hp/zones_filtered_size'] = self.config.environment.constrained_space.zones_filtered_size

    def close_runner(self):
        super().close_runner()
        self.run_stats_manager.close()
        clean_log_folder(run_code=self.run_code, run_config=self.config)
        del self.logger
