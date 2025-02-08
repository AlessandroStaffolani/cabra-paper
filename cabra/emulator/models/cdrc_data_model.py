import json
import os.path
from datetime import datetime, timedelta
from logging import Logger
from typing import List, Optional, Dict, Any, Union

import pandas as pd
from numpy.random import RandomState

from cabra import ROOT_DIR
from cabra.common.data_structure import RunMode, Done
from cabra.common.filesystem import create_directory_from_filepath
from cabra.common.mpi.mpi_tools import proc_id, num_procs
from cabra.core.step import Step
from cabra.core.step_data import StepData, StepDataEntry, StepWeather, NodeStepData, NodeStepDataCDRC
from cabra.emulator import EmulatorConfig
from cabra.emulator.data_structure import ModelTypes
from cabra.emulator.models.abstract import AbstractModel
from cabra.environment.node import Node


class CDRCDataModel(AbstractModel):

    def __init__(
            self,
            nodes: List[Node],
            random_state: RandomState,
            random_seed: int,
            emulator_configuration: Optional[EmulatorConfig] = None,
            log: Optional[Logger] = None,
            disable_log: bool = False,
            run_mode: RunMode = RunMode.Train,
            **kwargs
    ):
        self.rank: int = proc_id()
        self.processes: int = num_procs()
        self.is_root_process: bool = self.rank == 0
        self.use_mpi: bool = self.processes > 1 and run_mode == RunMode.Train
        super().__init__(
            nodes, model_name=ModelTypes.RealData,
            random_state=random_state, random_seed=random_seed,
            emulator_configuration=emulator_configuration, log=log, disable_log=disable_log, run_mode=run_mode)
        self.eval_every_week: bool = self.emulator_config.model.cdrc_data_model.eval_every_week
        self.training_dataset_path: str = self.emulator_config.model.cdrc_data_model.get_training_dataset_path()
        self.evaluation_dataset_path: str = self.emulator_config.model.cdrc_data_model.get_evaluation_dataset_path()
        self.nodes_path: str = self.emulator_config.model.cdrc_data_model.get_nodes_path()
        # self.weather_info_path: str = self.emulator_config.model.cdrc_data_model.get_weather_info_path()
        self.update_frequency: Step = self.emulator_config.model.cdrc_data_model.update_frequency
        self.episode_length: Step = self.emulator_config.model.cdrc_data_model.episode_length
        self.store_demand_stats = self.emulator_config.model.cdrc_data_model.store_demand_stats

        self.dataset: Dict[str, dict] = {}

        self.last_demand: Optional[StepData] = None

        self._load_dataset()
        self.current_index: int = self._set_initial_index()

        self.current_date: datetime = datetime.fromisoformat(self.dataset[str(self.current_index)]['date'])
        if self.episode_length is not None:
            self.next_done_date: datetime = self.current_date + timedelta(seconds=self.episode_length.total_steps)
        last_index = list(self.dataset.keys())[-1]
        self.max_date: datetime = datetime.fromisoformat(self.dataset[str(last_index)]['date'])
        self.days_evaluated = 0
        self.last_day = None

    def _set_initial_index(self):
        if self.use_mpi:
            dataset_chunk_size = len(self.dataset) // self.processes
            return dataset_chunk_size * self.rank
        else:
            return 0

    def _load_dataset(self):
        if self.run_mode == RunMode.Train:
            path = self.training_dataset_path
        elif self.run_mode == RunMode.Eval:
            path = self.evaluation_dataset_path
        else:
            raise AttributeError(f'Unsupported run_mode {self.run_mode}')
        with open(path, 'r') as f:
            self.dataset = json.load(f)

    def _predict_demand(self, step: Step) -> StepData:
        raise NotImplementedError('predict demand is not implemented yet')

    def _update_data(self, step: Step) -> StepData:
        index_updated = False
        if step.total_steps > 0 and step.total_steps % self.update_frequency.total_steps == 0:
            # we update the index, we load the new demand, and we update the next_change_step
            self.current_index += 1
            self.current_date += timedelta(seconds=self.update_frequency.total_steps)
            index_updated = True
        # we use the current index for loading the current demand, and we return it
        return self.get_step_data_internal(self.current_index, step, index_updated)

    def _update_nodes_capacity(
            self,
            node_name: str,
            node_step_data: Union[NodeStepData, NodeStepDataCDRC],
            nodes_mapping_to_update: Dict[str, Node]
    ) -> int:
        env_shortages = 0
        add_node = nodes_mapping_to_update[node_name]
        quantity = node_step_data.bikes
        if quantity != 0:
            if quantity > 0:
                if add_node.allocation_possible(quantity):
                    add_node.allocate(quantity)
                else:
                    shortages = quantity - add_node.empty_slots
                    if add_node.empty_slots > 0:
                        add_node.allocate(add_node.empty_slots)
                    env_shortages += shortages
            else:
                if add_node.removal_possible(abs(quantity)):
                    add_node.remove(abs(quantity))
                else:
                    shortages = abs(quantity) - add_node.bikes
                    if add_node.bikes > 0:
                        add_node.remove(add_node.bikes)
                    env_shortages += shortages
        return env_shortages

    def get_step_data_internal(
            self,
            index: int,
            step: Step,
            index_updated: bool,
            ignore_demand: bool = False
    ) -> StepData:
        if not index_updated and self.last_demand:
            return self.last_demand
        else:
            data = self.dataset[str(index)]
            step_weather = data['weather']
            demand = StepData(step, datetime.fromisoformat(data['date']))
            for node in self.nodes:
                if node.name in data['stations']:
                    bikes = data['stations'][node.name]
                else:
                    bikes = 0
                demand.populate_cdrc_data(node.name, bikes)
                if self.store_demand_stats:
                    self.store_node_stats_data(index, step, node, bikes)
            demand.data_index = index
            if not ignore_demand:
                self.last_demand = demand
            return demand

    def is_done(self, step: Step) -> Done:
        if self.eval_every_week:
            if self.last_day is not None and self.last_day != step.week_day:
                self.days_evaluated += 1
            self.last_day = step.week_day

        if str(self.current_index) not in self.dataset:
            return Done.Done
        if step.total_steps > 0 and step.total_steps % self.update_frequency.total_steps == 0:
            if self.eval_every_week and self.run_mode == RunMode.Eval:
                if (str(self.current_index + 1) not in self.dataset
                        or (self.days_evaluated > 0 and self.days_evaluated % 7 == 0)):
                    return Done.Done
            else:
                if str(self.current_index + 1) not in self.dataset:
                    return Done.Done
        if self.episode_length is not None and self.run_mode == RunMode.Train:
            if self.current_date >= self.next_done_date or self.current_date >= self.max_date:
                self.next_done_date += timedelta(seconds=self.episode_length.total_steps)
                return Done.VirtualDone

        return Done.NotDone

    def reset(self, reset_nodes=True):
        super().reset(reset_nodes)
        if self.eval_every_week and self.run_mode == RunMode.Eval:
            self.last_day = None
            self.days_evaluated = 0
            if str(self.current_index + 1) not in self.dataset:
                self.current_index = 0
                self.current_date: datetime = datetime.fromisoformat(self.dataset[str(self.current_index)]['date'])
        else:
            if self.episode_length is not None and self.run_mode == RunMode.Train:
                if self.next_done_date >= self.max_date or str(self.current_index) not in self.dataset:
                    # it is a full reset, otherwise keep going, we simply terminated one virtual episode
                    self.current_index = 0
                    self.current_date: datetime = datetime.fromisoformat(self.dataset[str(self.current_index)]['date'])
                    if self.episode_length is not None:
                        self.next_done_date: datetime = self.current_date + timedelta(
                            seconds=self.episode_length.total_steps)
            else:
                self.current_index = 0
                self.current_date: datetime = datetime.fromisoformat(self.dataset[str(self.current_index)]['date'])

    def store_node_stats_data(self, index: int, step: Step, node: Node, bikes: int):
        dataset_name = self.get_dataset_name()
        save_path = os.path.join(ROOT_DIR, f'data/datasets/{dataset_name}-dataset_stats.csv')
        create_directory_from_filepath(save_path)
        node_avail = node.bikes
        is_shortage = False
        if bikes != 0:
            if bikes > 0:
                node_avail += bikes
                if not node.allocation_possible(bikes):
                    is_shortage = True
            else:
                node_avail -= bikes
                if not node.removal_possible(abs(bikes)):
                    is_shortage = True
        data = {
            'index': index,
            'step': step.to_str(),
            'node': node.name,
            'availability': node_avail,
            'new_bikes': bikes,
            'causes_shortage': is_shortage
        }
        df = pd.DataFrame(data=data, index=[0])
        df.to_csv(save_path, header=not os.path.exists(save_path), index=False, mode='a')

    def get_dataset_name(self) -> str:
        eval_dataset_path = self.emulator_config.model.cdrc_data_model.get_evaluation_dataset_path()
        if 'dublin' in eval_dataset_path:
            return 'dublin'
        if 'london' in eval_dataset_path:
            return 'london'
        if 'paris' in eval_dataset_path:
            return 'paris'
        if 'newyork' in eval_dataset_path:
            return 'newyork'
