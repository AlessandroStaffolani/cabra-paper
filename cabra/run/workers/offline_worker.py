import os.path
import time
from typing import Optional, Tuple
from uuid import uuid4

from cabra import logger, SingleRunConfig
from cabra.common.filesystem import create_directory
from cabra.common.offline_queue import SupportedPatterns, OfflineRunsConfigQueue
from cabra.run.run_scheduler import CONFIG_BASE_PATH
from cabra.run.runner.offline_runner import OfflineRunner


class OfflineWorker:

    def __init__(
            self,
            queue_name: str,
            pattern: SupportedPatterns = SupportedPatterns.YAML,
            stop_empty: bool = False,
            max_runs: Optional[int] = None,
    ):
        self.queue_name = queue_name
        self.queue_path: str = os.path.join(CONFIG_BASE_PATH, self.queue_name)
        self.max_runs: Optional[int] = max_runs
        self.workers_folder_path: str = os.path.join(os.getenv('DATA_BASE_DIR'), 'workers')
        self.workers_manager_path: str = os.path.join(self.workers_folder_path, 'manager')
        self.workers_status_path: str = os.path.join(self.workers_folder_path, 'run_status')
        self.worker_code: str = f'worker-{str(uuid4())}'
        self.queue: OfflineRunsConfigQueue = OfflineRunsConfigQueue(
            queue_path=self.queue_path,
            pattern=pattern
        )
        self.stop_empty: bool = stop_empty
        self.stop: bool = False

        self._init_workers_folder()

        self.completed_runs = 0

    @property
    def stop_file_path(self):
        return os.path.join(self.workers_manager_path, self.worker_code)

    @property
    def worker_pid_file(self):
        return os.path.join(self.workers_manager_path, f'{self.worker_code}_pid')

    def _init_workers_folder(self):
        create_directory(self.workers_folder_path)
        create_directory(self.workers_manager_path)
        create_directory(self.workers_status_path)
        with open(self.worker_pid_file, 'w') as f:
            f.write(str(os.getpid()))

    def _check_requested_stop(self):
        if os.path.exists(self.stop_file_path):
            self.stop = True
            os.remove(self.stop_file_path)
            logger.info(f'Worker {self.worker_code} requested to stop')
        if self.max_runs is not None and self.completed_runs >= self.max_runs:
            self.stop = True
            logger.info(f'Worker {self.worker_code} completed {self.completed_runs} runs and it has been asked to stop')

    def _get_next_run_config(self) -> Optional[Tuple[Optional[str], Optional[SingleRunConfig], Optional[int]]]:
        run_code, run_config, run_index = self.queue.pop()
        if run_config is None and self.stop_empty:
            self.stop = True
            return None, None, None
        else:
            return run_code, run_config, run_index

    def _start_run(self, run_code: str, run_config: SingleRunConfig):
        logger.info(f'Worker {self.worker_code} is starting run {run_code}')
        runner = OfflineRunner(run_code=run_code, config=run_config, run_worker=self.worker_code)
        runner.run()
        self.completed_runs += 1

    def start(self):
        message = f'Started offline worker {self.worker_code} ' \
                  f'listening on queue "{self.queue_name}"'
        if self.max_runs is not None:
            message += f' - max runs limit: {self.max_runs}'
        logger.info(message)

        while not self.stop:
            next_run_code, next_run_config, next_run_index = self._get_next_run_config()
            if next_run_config is not None:
                # start run
                logger.info(f'Starting run with index: {next_run_index}')
                self._start_run(run_code=next_run_code, run_config=next_run_config)
            self._check_requested_stop()
            time.sleep(1)

        self.stop_worker()

    def stop_worker(self):
        logger.info(f'Worker {self.worker_code} is stopping')
        os.remove(self.worker_pid_file)

