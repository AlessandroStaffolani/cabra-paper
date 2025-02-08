import os
import socket
from logging import Logger
from typing import Optional, List

from cabra import SingleRunConfig, ROOT_DIR
from cabra.common.data_structure import RunMode
from cabra.common.object_handler import SaverMode
from cabra.run import RunStatus
from cabra.run.remote import MongoRunWrapper, PersistenceInfo, RedisRunWrapper
from cabra.run.run_stats_manager import RunStatsManager
from cabra.run.runner.base_runner import BaseRunner
from cabra.run.runner.utils import get_run_logger, complete_training_run, \
    clean_log_folder


class RemoteRunner(BaseRunner):

    def __init__(self, run_code: str):
        self.run_code: str = run_code.split('_')[0]
        self.run_worker: str = run_code.split('_')[1] if len(run_code.split('_')) > 1 else None
        mongo_host = None
        mongo_port = None

        self.mongo: MongoRunWrapper = MongoRunWrapper(host=mongo_host, port=mongo_port)
        run_info = self.mongo.get_by_run_code(run_code=self.run_code)
        if run_info is None:
            raise AttributeError(f'No run with code {run_code} exists')
        self.run_status = RunStatus(run_info['status'])
        if self.run_status == RunStatus.RUNNING or self.run_status == RunStatus.COMPLETED \
                or self.run_status == RunStatus.ERROR:
            raise AttributeError(
                f'The run with code {run_code} can not be started because the status is {self.run_status}')
        self.config: SingleRunConfig = SingleRunConfig(root_dir=ROOT_DIR, **run_info['config'])
        self.logger: Logger = get_run_logger(self.run_code, self.config)

        super(RemoteRunner, self).__init__(run_code=self.run_code, config=self.config, log=self.logger)

        self.redis: Optional[RedisRunWrapper] = None
        self.run_stats_manager: Optional[RunStatsManager] = None
        self.result_path: Optional[PersistenceInfo] = None
        self.iteration_checkpoint: List[int] = []
        self.last_batches_results: List[int] = []
        self.best_batch_results: Optional[float] = None
        self._init()

    def _init(self):
        try:
            if not self.initialized:
                self._init_seeds()
                self._init_env()
                self._init_agent()
                self._init_trucks_wrapper()
                self._init_eval_pool()
                self._init_run_stats_manager()
                self._init_redis()
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
        host = socket.gethostbyname(socket.gethostname())
        if self.config.saver.mode == SaverMode.Minio:
            host = self.object_handler.endpoint
        self.result_path: PersistenceInfo = PersistenceInfo(
            path=self.run_stats_manager.save_folder,
            save_mode=self.config.saver.mode,
            host=host
        )

    def _init_redis(self):
        if self.config.redis.enabled:
            redis_host = os.getenv('REDIS_HOST')
            redis_port = int(os.getenv('REDIS_PORT'))
            self.redis: RedisRunWrapper = RedisRunWrapper(
                run_code=self.run_code,
                run_total_steps=self.training_steps,
                logger=self.logger,
                config=self.config,
                host=redis_host,
                port=redis_port
            )

    def _update_redis_stats(self, current_step: int):
        if self.redis is not None:
            run_stats = self.stats_tracker.redis_stats(get_max_and_last=False)
            run_stats['evaluation_counter'] = self.completed_eval_counter
            if self.best_eval_performance is not None:
                run_stats['best_evaluation_performance'] = self.best_eval_performance
                run_stats['best_evaluation_performance_iteration'] = self.best_eval_performance_iteration
            self.redis.add_stats(
                run_worker=self.run_worker,
                current_step=current_step,
                status=RunStatus.RUNNING,
                run_stats=run_stats,
                agent_name=self.agent.name,
                zone_agent_name=self.zone_agent.name,
                multi_run_code=self.config.multi_run.multi_run_code,
                multi_run_params=self.config.multi_run.multi_run_params
            )

    def pre_run_start_callback(self, training_steps: int, log_train_info: bool = True, log_eval_info: bool = False):
        super(RemoteRunner, self).pre_run_start_callback(training_steps, log_train_info, log_eval_info)
        self.mongo.update_scheduled_run(self.run_code, status=RunStatus.RUNNING)
        if self.redis is not None:
            self.redis.add_stats(status=RunStatus.RUNNING,
                                 run_worker=self.run_worker,
                                 agent_name=self.agent.name,
                                 zone_agent_name=self.zone_agent.name,
                                 current_step=0, run_stats={})

    def after_train_step_callback(self, is_on_policy: bool, iteration: int):
        super(RemoteRunner, self).after_train_step_callback(is_on_policy, iteration)
        if iteration % self.config.run.info_frequency == 0:
            self._update_redis_stats(iteration)

    def save_agent_model(self, agent_model, filename: str):
        if self.run_mode == RunMode.Train:
            self.run_stats_manager.save_agent(
                agent_model, save_agent_state=self.agent.save_agent_state, filename=filename)
            self.logger.info(f'Save agent model {filename}')

    def after_run_callback(self):
        run_performance = super(RemoteRunner, self).after_run_callback()
        self._save_evaluation_data()
        self._after_run(run_performance)

    def _save_evaluation_data(self):
        prefix = RunMode.Eval.value
        if self.agent.requires_evaluation and self.agent.mode == RunMode.Train:
            reward = self.stats_tracker.get(f'{prefix}/avg/reward')
            for iteration in range(len(reward)):
                evaluation_data = {
                    f'{prefix}/avg/reward': self.stats_tracker.get(f'{prefix}/avg/reward')[iteration],
                    f'{prefix}/avg/cost': self.stats_tracker.get(f'{prefix}/avg/cost')[iteration],
                    f'{prefix}/avg/shortages': self.stats_tracker.get(f'{prefix}/avg/shortages')[iteration],
                    f'{prefix}/avg/env_shortages': self.stats_tracker.get(f'{prefix}/avg/env_shortages')[iteration],
                    f'{prefix}/avg/solved_steps': self.stats_tracker.get(f'{prefix}/avg/solved_steps')[
                        iteration],
                }
                for key, value in evaluation_data.items():
                    if hasattr(value, 'item'):
                        value = value.item()
                    evaluation_data[key] = value

                self.mongo.add_evaluation_run(
                    run_code=self.run_code,
                    iteration=iteration + 1,
                    status=RunStatus.COMPLETED,
                    evaluation_data=evaluation_data,
                    steps=self.config.run.evaluation_steps
                )

    def _after_run(self, run_performance: Optional[float]):
        self.run_stats_manager.save_stats(agent_state=None, save_agent_state=self.agent.save_agent_state)
        # set as completed the run on redis
        if self.redis is not None:
            redis_stats = self.stats_tracker.redis_stats()
            if self.run_mode == RunMode.Train:
                redis_stats = self.stats_tracker.redis_stats()
            self.redis.add_stats(
                current_step=self.env.time_step.current_step,
                status=RunStatus.COMPLETED,
                run_worker=self.run_worker,
                agent_name=self.agent.name,
                zone_agent_name=self.zone_agent.name,
                run_stats=redis_stats
            )
        # set as completed the run on mongo
        db_stats = self.stats_tracker.db_stats()
        db_stats[f'{RunMode.Eval.value}/evaluations_counter'] = self.completed_eval_counter
        if self.best_eval_performance is not None:
            db_stats[f'{RunMode.Eval.value}/best/performance'] = self.best_eval_performance
            db_stats[
                f'{RunMode.Eval.value}/best/performance_iteration'] = self.best_eval_performance_iteration
        data_folder = None
        self.mongo.update_scheduled_run(
            run_code=self.run_code,
            status=RunStatus.COMPLETED,
            result_path=self.result_path,
            total_steps=self.env.time_step.current_step.total_steps,
            run_stats=db_stats,
            run_performance=run_performance
        )
        complete_training_run(
            run_code=self.run_code,
            mongo=self.mongo,
            config=self.config,
            redis=self.redis
        )

    def close_runner(self):
        super().close_runner()
        self.run_stats_manager.close()
        if self.redis is not None:
            self.redis.close()
        self.mongo.close()
        clean_log_folder(run_code=self.run_code, run_config=self.config)
        del self.logger
