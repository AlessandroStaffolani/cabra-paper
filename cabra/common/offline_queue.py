import errno
import os
from glob import glob
from typing import List, Optional, Tuple

import yaml

from cabra import SingleRunConfig, ROOT_DIR
from cabra.common.enum_utils import ExtendedEnum


def silent_remove(filename):
    try:
        os.remove(filename)
    except OSError as e:  # this would be "except OSError, e:" before Python 2.6
        if e.errno != errno.ENOENT:  # errno.ENOENT = no such file or directory
            raise  # re-raise exception if a different error occurred


class SupportedPatterns(str, ExtendedEnum):
    YAML = 'yaml'
    # JSON = 'json'


def sort_glob_result(file: str):
    try:
        filename = file.split('/')[-1]
        index = int(filename.split('_')[0])
        return index
    except Exception:
        try:
            return file.split('/')[-1]
        except Exception:
            return file


class OfflineRunsConfigQueue:

    def __init__(
            self,
            queue_path: str,
            pattern: SupportedPatterns = SupportedPatterns.YAML
    ):
        self.pattern: SupportedPatterns = pattern
        self.queue: str = queue_path
        self.files_queue: List[str] = []
        self.sync_files_queue()

    def sync_files_queue(self):
        self.files_queue = sorted(
            glob(f'{self.queue}/*.{self.pattern.value}'),
            key=lambda file: sort_glob_result(file)
        )

    def pop(self) -> Tuple[Optional[str], Optional[SingleRunConfig], Optional[int]]:
        self.sync_files_queue()
        if len(self.files_queue) == 0:
            return None, None, None
        next_run = self.files_queue.pop(0)
        try:
            with open(next_run, 'r') as file:
                if self.pattern == SupportedPatterns.YAML:
                    content = yaml.safe_load(file.read())
                # elif self.pattern == SupportedPatterns.JSON:
                #     content = json.load(file)
                else:
                    raise AttributeError(f'Pattern {self.pattern} not supported')
                os.remove(next_run)
                self.sync_files_queue()
                config = SingleRunConfig(ROOT_DIR)
                config.set_configs(**content)
                run_code = next_run.split('/')[-1].replace(f'.{self.pattern.value}', '')
                run_info = run_code.split('_')
                run_index = int(run_info[0])
                run_code = run_info[-1]
                return run_code, config, run_index

        except OSError as e:
            if e.errno == errno.ENOENT:  # errno.ENOENT = no such file or directory
                return self.pop()
            else:
                raise e


