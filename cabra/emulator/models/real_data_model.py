import json
from datetime import datetime, timedelta
from logging import Logger
from typing import List, Optional, Dict, Any

from numpy.random import RandomState

from cabra.common.data_structure import RunMode, Done
from cabra.common.mpi.mpi_tools import proc_id, num_procs
from cabra.core.step import Step
from cabra.core.step_data import StepData, StepDataEntry, StepWeather
from cabra.emulator import EmulatorConfig
from cabra.emulator.data_structure import ModelTypes
from cabra.emulator.models.abstract import AbstractModel
from cabra.environment.node import Node


class RealDataModel(AbstractModel):

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
        super(RealDataModel, self).__init__(
            nodes, model_name=ModelTypes.RealData,
            random_state=random_state, random_seed=random_seed,
            emulator_configuration=emulator_configuration, log=log, disable_log=disable_log, run_mode=run_mode)
        self.training_dataset_path: str = self.emulator_config.model.realdata_model.get_training_dataset_path()
        self.evaluation_dataset_path: str = self.emulator_config.model.realdata_model.get_evaluation_dataset_path()
        self.nodes_path: str = self.emulator_config.model.realdata_model.get_nodes_path()
        self.weather_info_path: str = self.emulator_config.model.realdata_model.get_weather_info_path()
        self.update_frequency: Step = self.emulator_config.model.realdata_model.update_frequency
        self.episode_length: Step = self.emulator_config.model.realdata_model.episode_length

        self.weather_info: Dict[str, Any] = self._load_weather_info()
        self.dataset: Dict[str, dict] = {}

        self.last_demand: Optional[StepData] = None

        self._load_dataset()
        self.current_index: int = self._set_initial_index()

        self.current_date: datetime = datetime.fromisoformat(self.dataset[str(self.current_index)]['date'])
        if self.episode_length is not None:
            self.next_done_date: datetime = self.current_date + timedelta(seconds=self.episode_length.total_steps)
        last_index = list(self.dataset.keys())[-1]
        self.max_date: datetime = datetime.fromisoformat(self.dataset[str(last_index)]['date'])

    def _set_initial_index(self):
        if self.use_mpi:
            dataset_chunk_size = len(self.dataset) // self.processes
            return dataset_chunk_size * self.rank
        else:
            return 0

    def _load_weather_info(self):
        with open(self.weather_info_path, 'r') as f:
            return json.load(f)

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
        # we use the current index + 1 for loading the next demand, and we return it
        raise NotImplementedError('predict demand is not implemented yet')
        # return self._get_step_data(self.current_index + 1, step)

    def _update_data(self, step: Step) -> StepData:
        index_updated = False
        if step.total_steps > 0 and step.total_steps % self.update_frequency.total_steps == 0:
            # we update the index, we load the new demand, and we update the next_change_step
            self.current_index += 1
            self.current_date += timedelta(seconds=self.update_frequency.total_steps)
            index_updated = True
        # we use the current index for loading the current demand, and we return it
        return self.get_step_data_internal(self.current_index, step, index_updated)

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
            demand = StepData(step, datetime.fromisoformat(data['date']),
                              weather=self._prepare_weather_data(step_weather))
            for node in self.nodes:
                if node.name in data['stations']:
                    node_data = data['stations'][node.name]
                    demand.populate_data(node.name, node_data['ended'], node_data['started']['out_interval'])
                else:
                    demand.populate_data(node.name, {}, 0)
            demand.data_index = index
            if not ignore_demand:
                self.last_demand = demand
            return demand

    def _prepare_weather_data(self, weather_data) -> StepWeather:
        return StepWeather(
            condition=self.weather_info['mapping'][weather_data['condition']],
            temperature=weather_data['temperature'],
            wind_speed=weather_data['wind_speed']
        )

    def is_done(self, step: Step) -> Done:
        if str(self.current_index) not in self.dataset:
            return Done.Done
        if step.total_steps > 0 and step.total_steps % self.update_frequency.total_steps == 0:
            if str(self.current_index + 1) not in self.dataset:
                return Done.Done
        if self.episode_length is not None and self.run_mode == RunMode.Train:
            if self.current_date >= self.next_done_date or self.current_date >= self.max_date:
                self.next_done_date += timedelta(seconds=self.episode_length.total_steps)
                return Done.VirtualDone
        return Done.NotDone

    def reset(self, reset_nodes=True):
        super(RealDataModel, self).reset(reset_nodes)
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
