import os
import sys
from typing import Union, List
from uuid import uuid4

from dotenv import load_dotenv

from cabra import single_run_config, multi_run_config, SingleRunConfig, ROOT_DIR
from cabra.common.controller import Controller
from cabra.common.mpi.mpi_tools import mpi_fork
from cabra.common.object_handler import create_object_handler
from cabra.common.object_handler.base_handler import ObjectHandler
from cabra.common.object_handler.minio_handler import MinioObjectHandler
from cabra.environment import logger
from cabra.run.run_scheduler import RunScheduler
from cabra.run.runner import RemoteRunner, TestRunner
from cabra.run.workers.run_worker import MPI_RUNNER_SCRIPT


class RunnerController(Controller):

    def __init__(self):
        super(RunnerController, self).__init__('RunnerController')
        self._add_action('train', self.train)
        self._add_action('test-runs', self.test_runs)
        self.object_handler: Union[ObjectHandler, MinioObjectHandler] = create_object_handler(
            logger=logger,
            enabled=single_run_config.saver.enabled,
            mode=single_run_config.saver.mode,
            base_path=single_run_config.saver.get_base_path(),
            default_bucket=single_run_config.saver.default_bucket
        )

    def _single_train(self):
        run_code: str = str(uuid4())
        self.run_scheduler: RunScheduler = RunScheduler()
        self.run_scheduler.schedule_run(run_code, config=single_run_config)
        runner = RemoteRunner(
            run_code=run_code
        )
        runner.run()

    def train(self, **kwargs):
        self._single_train()

    def test_runs(self, multi: bool = False, use_mpi: bool = False, processes: int = 4, **kwargs):
        if use_mpi:
            load_dotenv(dotenv_path=os.path.join(ROOT_DIR, '.env.local'), override=True)
            if 'REDIS_PASSWORD' in os.environ:
                del os.environ['REDIS_PASSWORD']
            run_scheduler: RunScheduler = RunScheduler()
            run_code = str(uuid4())
            run_scheduler.schedule_run(run_code=run_code, config=single_run_config, queue_name='test_run')
            logger.info(f'Scheduled test run with run_code: {run_code}')
            logger.info(f'Preparing MPI runner with {processes} parallel workers')
            mpi_fork(n=processes, run_code=run_code, script_name=os.path.join(ROOT_DIR, MPI_RUNNER_SCRIPT),
                     exit_on_end=False, is_test_run=True)
            run_scheduler.redis_queue.redis_connection.lrem('test_run', 0, run_code)
            run_scheduler.redis_queue.redis_connection.delete(f'{run_code}_stats')
            logger.info(f'Test run completed and removed run_code {run_code} from test_run queue ')
            sys.exit()
        if multi:
            runs: List[SingleRunConfig] = multi_run_config.generate_runs_config()
            for run_config in runs:
                runner = TestRunner(run_code=str(uuid4()), config=run_config)
                runner.run()
        else:
            run_code: str = str(uuid4())
            runner = TestRunner(run_code=run_code, config=single_run_config)
            runner.run()
