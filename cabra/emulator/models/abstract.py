import json
import os
from abc import abstractmethod
from glob import glob
from logging import Logger
from typing import List, Optional, Any, Dict, Tuple, Union

from numpy.random import RandomState

from cabra.common.data_structure import RunMode, Done
from cabra.core.step import Step
from cabra.core.step_data import StepDataEntry, StepData, NodeStepData, NodeStepDataCDRC
from cabra.emulator import logger, EmulatorConfig, emulator_config
from cabra.emulator.data_structure import ModelTypes
from cabra.environment.node import Node


def parse_entry(entry: Dict[str, dict]) -> List[StepDataEntry]:
    entries: List[StepDataEntry] = []
    step = Step(**entry['step'])
    for res_name, res_data in entry['data'].items():
        for bs_data in res_data:
            bs_name = list(bs_data.keys())[0]
            bs_value = list(bs_data.values())[0]
            entries.append(StepDataEntry(value=bs_value, node=bs_name, step=step))
    return entries


def parse_step_data(entry: Dict[str, dict]) -> StepData:
    entries = parse_entry(entry)
    return StepData(*entries)


class AbstractModel:

    def __init__(
            self,
            nodes: List[Node],
            model_name: ModelTypes,
            random_state: RandomState,
            random_seed: int,
            emulator_configuration: Optional[EmulatorConfig] = None,
            log: Optional[Logger] = None,
            disable_log: bool = False,
            run_mode: RunMode = RunMode.Train,
    ):
        self.emulator_config: EmulatorConfig = emulator_configuration if emulator_configuration is not None \
            else emulator_config
        self.logger: Logger = log if log is not None else logger
        self.run_mode: RunMode = run_mode
        self.random: RandomState = random_state
        self.random_seed: int = random_seed
        self.nodes: List[Node] = nodes
        self.nodes_mapping: Dict[str, Node] = {n.name: n for n in self.nodes}
        self._name: ModelTypes = model_name
        self.disable_log: bool = disable_log
        self.use_predictions: bool = self.emulator_config.model.use_predictions
        self.current_demand: Optional[StepData] = None
        self.predicted_demand: Optional[StepData] = None
        self.eval_every_week: bool = False

    def model_name(self):
        return self._name

    def get_step_demand(self, step: Step) -> StepData:
        return self._update_data(step)

    @abstractmethod
    def _predict_demand(self, step: Step) -> StepData:
        pass

    def predict_demand(self, step: Step):
        self.predicted_demand = self._predict_demand(step)

    def update_data(self, current_step: Step, next_step: Optional[Step] = None) -> Tuple[StepData, int]:
        data: StepData = self.get_step_demand(current_step)
        self.current_demand = data
        invalid_movements = self.update_nodes_capacity(self.current_demand)

        if self.use_predictions and next_step is not None:
            self.predict_demand(next_step)
            self.update_nodes_capacity(self.predicted_demand, is_prediction=True)

        return data, invalid_movements

    def update_prediction_only(self, next_step: Optional[Step] = None):
        if next_step is not None:
            self.predict_demand(next_step)
            self.update_nodes_capacity(self.predicted_demand, is_prediction=True)

    def update_nodes_capacity(
            self,
            demand: StepData,
            is_prediction=False,
            nodes_mapping: Optional[Dict[str, Node]] = None
    ) -> int:
        env_shortages = 0
        node_step_data: NodeStepData
        nodes_mapping_to_update = self.nodes_mapping if nodes_mapping is None else nodes_mapping
        for node_name, node_step_data in demand.items():
            env_shortages += self._update_nodes_capacity(node_name, node_step_data, nodes_mapping_to_update)
        return env_shortages

    def _update_nodes_capacity(
            self,
            node_name: str,
            node_step_data: Union[NodeStepData, NodeStepDataCDRC],
            nodes_mapping_to_update: Dict[str, Node]
    ) -> int:
        env_shortages = 0
        add_node = nodes_mapping_to_update[node_name]
        add_node.set_ongoing_trips(node_step_data.started_out_interval)
        if len(node_step_data.ended) > 0:
            for remove_node_name, info in node_step_data.ended.items():
                remove_node = nodes_mapping_to_update[remove_node_name]
                quantity = info['n_bikes']
                if add_node.allocation_possible(quantity) and remove_node.removal_possible(quantity):
                    # it is possible, we do the movement
                    remove_node.remove(quantity)
                    add_node.allocate(quantity)
                else:
                    # we do nothing, it is invalid, and we increase the counter
                    env_shortages += 1
        return env_shortages

    def update_prediction(self):
        pass

    def reset(self, reset_nodes=True):
        if reset_nodes:
            for node in self.nodes:
                node.reset()

    @abstractmethod
    def _update_data(self, step: Step) -> StepData:
        pass

    def is_done(self, step: Step) -> Done:
        return Done.NotDone
