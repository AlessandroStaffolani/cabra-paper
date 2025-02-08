from logging import Logger
from typing import Optional
from uuid import uuid4

from cabra import SingleRunConfig
from cabra.run.runner.base_runner import BaseRunner


class TestRunner(BaseRunner):

    def __init__(
            self,
            run_code: Optional[str] = None,
            config: Optional[SingleRunConfig] = None,
            log: Optional[Logger] = None
    ):
        super(TestRunner, self).__init__(
            run_code=run_code if run_code is not None else str(uuid4()),
            config=config,
            log=log
        )

    def _print_action_history(self):
        action_history = self.agent.selected_actions
        self.logger.info('Action History')
        for sub_action, history in action_history.items():
            for action, count in history.items():
                self.logger.info(f'Sub-action: {sub_action.value} | action {action}: {count}')

    def _pre_close(self):
        self._print_action_history()
