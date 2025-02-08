import os
from glob import glob
from typing import List, Tuple, Optional
from uuid import uuid4

from cabra import SingleRunConfig, MultiRunConfig, ROOT_DIR
from cabra.common.config import ExportMode
from cabra.common.filesystem import create_directory
from cabra.common.offline_queue import SupportedPatterns
from cabra.common.print_utils import print_status
from cabra.common.remote.redis_wrapper import RedisQueue
from cabra.run.remote import MongoRunWrapper
from cabra.run.remote.redis_constants import REDIS_RUNS_QUEUE, REDIS_EVAL_RUNS_QUEUE

CONFIG_BASE_PATH = os.path.join(ROOT_DIR, 'config')


class RunScheduler:

    def __init__(self, is_offline: bool):
        self.mongo = None
        self.redis_queue: None
        if not is_offline:
            mongo_host = None
            mongo_port = None
            redis_host = os.getenv('REDIS_HOST')
            redis_port = int(os.getenv('REDIS_PORT'))
            self.mongo: MongoRunWrapper = MongoRunWrapper(host=mongo_host, port=mongo_port)
            self.redis_queue: RedisQueue = RedisQueue(
                host=redis_host,
                port=redis_port,
                db=int(os.getenv('REDIS_DB')),
                password=os.getenv('REDIS_PASSWORD')
            )

    def schedule_run(self, run_code: str, config: SingleRunConfig, is_eval_run=False,
                     queue_name: Optional[str] = None, schedule_locally: bool = False):
        if schedule_locally:
            self.schedule_locally(run_code, config, folder=queue_name, index=0)
        else:
            self.mongo.add_scheduled_run(run_code, run_config=config)
            queue = REDIS_RUNS_QUEUE if queue_name is None else queue_name
            if is_eval_run:
                queue = REDIS_EVAL_RUNS_QUEUE if queue_name is None else queue_name
            self.redis_queue.push(key=queue, value=run_code, allow_duplicates=False)

    def schedule_multi_runs(self, config: MultiRunConfig, is_eval_run=False,
                            queue_name: Optional[str] = None, schedule_locally: bool = False) -> Tuple[int, str]:
        runs: List[SingleRunConfig] = config.generate_runs_config()
        multi_run_code = runs[0].multi_run.multi_run_code
        total = len(runs)
        queue = REDIS_RUNS_QUEUE if queue_name is None else queue_name
        folder_path = os.path.join(CONFIG_BASE_PATH, queue)
        queue_len = 0
        if os.path.exists(folder_path):
            queue_len = len(glob(f'{folder_path}/*.{SupportedPatterns.YAML.value}'))

        for i, run_config in enumerate(runs):
            run_code = str(uuid4())
            if schedule_locally:
                self.schedule_locally(run_code, run_config, folder=queue_name, index=i + 1, queue_len=queue_len)
            else:
                self.schedule_run(run_code, run_config, is_eval_run=is_eval_run, queue_name=queue_name)
            print_status(current=i+1, total=total, pre_message=f'Runs scheduled for multi run: {multi_run_code}',
                         loading_len=40)
        print()
        return len(runs), multi_run_code

    def schedule_locally(
            self,
            run_code: str,
            run_config: SingleRunConfig,
            folder: str,
            index: int,
            queue_len: int = 0
    ):
        queue = REDIS_RUNS_QUEUE if folder is None else folder
        folder_path = os.path.join(CONFIG_BASE_PATH, queue)
        create_directory(folder_path)
        filename = f'{queue_len + index}_{run_code}.yaml'
        run_config.export(save_path=os.path.join(folder_path, filename), mode=ExportMode.YAML)

    def stop(self):
        pass
