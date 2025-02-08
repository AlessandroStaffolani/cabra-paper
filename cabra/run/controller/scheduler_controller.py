import re
from glob import glob
from typing import Optional
from uuid import uuid4

from cabra import single_run_config, multi_run_config, logger, ROOT_DIR, MultiRunConfig, \
    get_multi_run_config
from cabra.common.controller import Controller
from cabra.run.run_scheduler import RunScheduler


class SchedulerController(Controller):

    def __init__(self):
        super(SchedulerController, self).__init__('SchedulerController')
        self._add_action('one-run', self.schedule_single_run)
        self._add_action('multi-runs', self.schedule_multi_runs)

    def schedule_single_run(self, offline: bool = False, is_eval: bool = False,
                            queue: Optional[str] = None, schedule_locally: bool = False, **kwargs):
        run_code: str = str(uuid4())
        run_scheduler: RunScheduler = RunScheduler(offline)
        run_scheduler.schedule_run(run_code, config=single_run_config, is_eval_run=is_eval, queue_name=queue,
                                   schedule_locally=schedule_locally)
        logger.info(f'Scheduled a run with code {run_code}')
        run_scheduler.stop()

    def schedule_multi_runs(self, from_folder: Optional[str] = None,
                            offline: bool = False,
                            is_eval: bool = False, queue: Optional[str] = None,
                            schedule_locally: bool = False, **kwargs):
        run_scheduler: RunScheduler = RunScheduler(offline)
        if from_folder is not None:
            self._schedule_multi_runs_from_folder(run_scheduler, from_folder, is_eval=is_eval, queue=queue,
                                                  schedule_locally=schedule_locally)
        else:
            n_runs, multi_run_code = run_scheduler.schedule_multi_runs(config=multi_run_config,
                                                                       is_eval_run=is_eval,
                                                                       queue_name=queue,
                                                                       schedule_locally=schedule_locally)
            logger.info(f'Scheduled {n_runs} runs with multi-run-code: {multi_run_code}')
        run_scheduler.stop()

    def _schedule_multi_runs_from_folder(self, run_scheduler: RunScheduler, folder_path: str,
                                         is_eval: bool = False, queue: Optional[str] = None,
                                         schedule_locally: bool = False):
        path = folder_path
        if path[-1] == '/':
            path = path[:-1]
        runs_config_path = []
        runs_config_path += glob(f'{path}/*.yml')
        runs_config_path += glob(f'{path}/*yaml')

        def sort_if_path_begins_with_numbers(element):
            parts = element.split('/')
            last = parts[-1]
            int_pos = re.search('(\d+)', last)
            if int_pos is not None and int_pos.span()[0] == 0:
                return float(int_pos.group())
            return 0

        runs_config_path.sort(key=sort_if_path_begins_with_numbers)
        runs_scheduled = 0
        for run_path in runs_config_path:
            config: MultiRunConfig = get_multi_run_config(root_dir=ROOT_DIR, config_path=run_path)
            n_runs, multi_run_code = run_scheduler.schedule_multi_runs(config=config,
                                                                       is_eval_run=is_eval, queue_name=queue,
                                                                       schedule_locally=schedule_locally)
            logger.info(f'Scheduled {n_runs} runs with multi-run-code: {multi_run_code}')
            runs_scheduled += n_runs
        logger.info(f'Scheduled in total {runs_scheduled} runs')
