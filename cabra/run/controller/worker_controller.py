from typing import Optional

from cabra.common.controller import Controller
from cabra.run.remote.redis_constants import REDIS_RUNS_QUEUE
from cabra.run.workers.offline_worker import OfflineWorker
from cabra.run.workers.run_worker import WorkerType, Worker


class WorkerController(Controller):

    def __init__(self):
        super(WorkerController, self).__init__('WorkerController')
        self._add_action('offline-worker', self.offline_worker)
        self._add_action('run-worker', self.run_worker)

    def offline_worker(
            self,
            queue: Optional[str] = REDIS_RUNS_QUEUE,
            stop_empty: bool = False,
            max_runs: Optional[int] = None,
            **kwargs,
    ):
        worker = OfflineWorker(
            queue_name=queue,
            stop_empty=stop_empty,
            max_runs=max_runs
        )
        worker.start()

    def run_worker(self, processes: int,
                   queue: Optional[str] = None,
                   stop_empty: bool = False, use_mpi: bool = False, max_runs: Optional[int] = None, **kwargs):
        worker = Worker(
            processes=processes,
            worker_type=WorkerType.RunWorker,
            queue_name=queue,
            stop_empty=stop_empty,
            use_mpi=use_mpi,
            max_runs=max_runs
        )
        worker.start()

    def eval_run_worker(self, processes: int,
                        queue: Optional[str] = None, stop_empty: bool = False, **kwargs):
        worker = Worker(
            processes=processes,
            worker_type=WorkerType.EvalWorker,
            queue_name=queue,
            stop_empty=stop_empty
        )
        worker.start()
