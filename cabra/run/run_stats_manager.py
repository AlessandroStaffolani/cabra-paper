import os
from logging import Logger
from typing import List, Optional, Union, Any, Dict

import torch

from cabra import SingleRunConfig, ROOT_DIR
from cabra.common.data_structure import RunMode
from cabra.common.object_handler import get_save_folder_from_config, ObjectHandler, MinioObjectHandler
from cabra.common.stats_tracker import Tracker
from cabra.common.tensorboard_wrapper import TensorboardWrapper
from cabra.core.state import State
from cabra.environment.agent.abstract import AgentAbstract
from cabra.environment.node import Node


class RunStatsManager:

    def __init__(
            self,
            run_code: str,
            config: SingleRunConfig,
            nodes: List[Node],
            agent_name: str,
            total_steps: int,
            step_size: int,
            object_handler: Union[ObjectHandler, MinioObjectHandler],
            logger: Logger,
            save_folder: Optional[str] = None
    ):
        self.config: SingleRunConfig = config
        self.logger: Logger = logger
        self.run_code = run_code
        self.run_mode: RunMode = self.config.run.run_mode
        self.nodes: List[Node] = nodes
        self.stats_tracker: Optional[Tracker] = None
        self.object_handler: Union[ObjectHandler, MinioObjectHandler] = object_handler
        self.agent_name: str = agent_name
        self.total_steps: int = total_steps
        self.step_size: int = step_size
        self.cumulative_problem_solved: int = 0
        self.save_folder: Optional[str] = save_folder
        self.tensorboard: Optional[TensorboardWrapper] = None
        self._init()

    def _init(self):
        self._init_saver()

    def _init_saver_single_run(self):
        final_folder = get_save_folder_from_config(
            save_name=self.config.saver.save_name,
            save_prefix=self.config.saver.save_prefix,
            save_name_with_date=self.config.saver.save_name_with_date,
            save_name_with_uuid=self.config.saver.save_name_with_uuid,
            uuid=self.run_code
        )
        model_type_value = self.config.emulator.model.type
        self.save_folder: str = os.path.join(model_type_value,
                                             self.run_mode.value,
                                             self.agent_name, final_folder)

    def _init_saver_multi_run(self):
        multi_run_code = self.config.multi_run.multi_run_code
        if self.run_mode == RunMode.Train or self.run_mode == RunMode.Validation:
            final_folder = '_'.join([param.filename_key_val for param in self.config.multi_run.multi_run_params])
        elif self.run_mode == RunMode.Eval:
            final_folder = f'seed={self.config.random_seeds.evaluation[0]}_'
            final_folder += '_'.join([
                param.filename_key_val for param in self.config.multi_run.multi_run_params
                if 'seed' not in param.filename_key_val
            ])
            if multi_run_code.endswith('-evaluation'):
                multi_run_code = multi_run_code.replace('-evaluation', '')
        else:
            final_folder = ''
        model_type_value = self.config.emulator.model.type
        self.save_folder: str = os.path.join(
            multi_run_code,
            model_type_value,
            self.run_mode.value,
            self.agent_name,
            final_folder
        )

    def _init_saver(self):
        if self.config.multi_run.is_multi_run:
            self._init_saver_multi_run()
        else:
            self._init_saver_single_run()
        if self.config.saver.tensorboard.enabled:
            tensorboard_path = os.path.join(
                ROOT_DIR,
                self.config.saver.tensorboard.get_save_path(),
                self.save_folder
            )
            self.tensorboard: TensorboardWrapper = TensorboardWrapper(log_dir=tensorboard_path, logger=self.logger)
        if self.config.saver.stats_condensed:
            self.stats_tracker: Tracker = Tracker.init_condensed_tracker(
                run_code=self.run_code, config=self.config, run_mode=self.run_mode, tensorboard=self.tensorboard
            )
        else:
            self.stats_tracker: Tracker = Tracker.init_tracker(
                nodes=self.nodes,
                run_code=self.run_code, config=self.config, run_mode=self.run_mode, tensorboard=self.tensorboard
            )

    def save_stats(self, agent_state: Optional[Dict[str, Any]] = None, save_agent_state: bool = True):
        if self.config.saver.enabled:
            if agent_state is not None:
                self.logger.debug('Starting stats and agent saving procedure')
            data_to_save = self.stats_tracker.disk_stats()
            filename = 'full_data.json'
            message = f'Saved all the run stats into {os.path.join(self.save_folder, filename)}'
            self.object_handler.save(
                obj=data_to_save,
                path=self.save_folder,
                filename=filename,
                max_retries=3,
                wait_timeout=2,
                use_progress=True
            )
            self.logger.info(message)
        if self.config.saver.save_agent and agent_state is not None:
            self.save_agent(agent_state, save_agent_state)

    def save_agent(self, agent_state: Dict[str, Any], save_agent_state: bool, filename: str = 'agent_state.pth'):
        if self.config.saver.enabled and save_agent_state:
            self.object_handler.save_agent_model(
                agent_model=agent_state,
                filename=filename,
                path=self.save_folder,
                max_retries=3,
                wait_timeout=2,
                use_progress=True
            )

    def add_agent_net_graph(self, agent: AgentAbstract, state: State):
        if self.tensorboard is not None and self.config.saver.tensorboard.save_model_graph:
            agent_model = agent.get_model()
            if agent_model is not None:
                test_state = [
                    state.tolist(),
                    state.tolist(),
                    state.tolist(),
                    state.tolist(),
                ]
                self.tensorboard.add_graph(agent_model, input_to_model=torch.tensor(test_state, dtype=torch.float))

    def close(self):
        if self.tensorboard is not None:
            self.tensorboard.close()
