import json
import os
from logging import Logger
from multiprocessing.pool import ThreadPool
from typing import Optional, Dict, Union, List

from cabra import SingleRunConfig, single_run_config
from cabra.common.encoders import NumpyEncoder
from cabra.common.remote.redis_wrapper import RedisWrapper
from cabra.environment.agent import AgentType


class RedisRunWrapper:

    def __init__(
            self,
            run_code: str,
            run_total_steps: int,
            logger: Logger,
            host: str,
            port: int,
            config: Optional[SingleRunConfig] = None,
            val_run_code: Optional[str] = None,
    ):
        self.config: SingleRunConfig = config if config is not None else single_run_config
        self.logger: Logger = logger
        db = int(os.getenv('REDIS_DB'))
        self.redis: RedisWrapper = RedisWrapper(
            host=host,
            port=port,
            db=db
        )
        self.run_code: str = run_code
        self.val_run_code: Optional[str] = val_run_code
        self.run_total_steps: int = run_total_steps
        if not self.redis.check_connection():
            raise ValueError(
                f'Impossible to connect to redis using host "{self.redis.host} and port "{self.redis.port}')

        # redis run fields
        self._stats_key = f'{self.run_code}_stats'
        if self.val_run_code is not None:
            self._stats_key = f'{self.val_run_code}_stats'
        self.pool = ThreadPool(processes=2)

    def add_stats(
            self,
            status: str,
            current_step: int,
            run_worker: str,
            run_stats: Dict[str, Union[float, int]],
            agent_name: AgentType,
            zone_agent_name: AgentType,
            multi_run_code: Optional[str] = None,
            multi_run_params: List[dict] = []
    ):
        stats = {
            'status': status,
            'run_worker': run_worker,
            'current_iteration': current_step,
            'total_training_iterations': self.run_total_steps,
            'agent': agent_name.value,
            'zone_agent': zone_agent_name.value,
        }
        if multi_run_code is not None:
            stats['multi_run_code'] = multi_run_code
        if len(multi_run_params) > 0:
            for param_conf in multi_run_params:
                stats[param_conf['key']] = param_conf['value']
        if self.val_run_code is not None:
            stats['original_run_code'] = self.run_code
        for key, val in run_stats.items():
            stats[key] = val
        self.pool.apply_async(
            func=self._save_async,
            kwds={'key': self._stats_key, 'value': json.dumps(stats, cls=NumpyEncoder)},
            error_callback=lambda e: self.logger.exception(e)
        )

    def _save_async(self, key: str, value: str):
        self.redis.save(key, value)

    def add_scheduled_validation_run(self, val_run_code: str, queue_name: str):
        if self.val_run_code is None:
            return self.redis.redis_connection.rpush(queue_name, val_run_code)
        else:
            self.logger.warning('Called add_scheduled_validation_run from a validation run')

    def delete_single_run_stats(self, run_code: str):
        pattern = f'{run_code}_*stats'
        self.redis.delete_all(pattern)

    def close(self):
        self.pool.close()
        self.pool.join()
        self.redis.close()
