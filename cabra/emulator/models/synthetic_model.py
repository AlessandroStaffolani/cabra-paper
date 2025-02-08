from logging import Logger
from typing import List, Optional, Dict

import numpy as np
from numpy.random import RandomState

from cabra.common.data_structure import RunMode
from cabra.core.step import Step
from cabra.core.step_data import StepData, StepDataEntry
from cabra.emulator import EmulatorConfig
from cabra.emulator.data_structure import ModelTypes
from cabra.emulator.models.abstract import AbstractModel
from cabra.environment.node import Node


class SyntheticModel(AbstractModel):

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
        super(SyntheticModel, self).__init__(
            nodes, model_name=ModelTypes.SyntheticDataGenerator,
            random_state=random_state, random_seed=random_seed,
            emulator_configuration=emulator_configuration, log=log, disable_log=disable_log, run_mode=run_mode)
        self.movement_min: float = self.emulator_config.model.synthetic_model.movement_range[0]
        self.movement_max: float = self.emulator_config.model.synthetic_model.movement_range[1]

        self.gaussian_noise: Dict[str, float] = self.emulator_config.model.synthetic_model.gaussian_noise
        self.predicted_demand_clean: Optional[StepData] = None

        self.last_step_update: Optional[Step] = None

    def _generate_step_data(self, step: Step, prediction_noise: Optional[np.ndarray] = None) -> StepData:
        step_entries: List[StepDataEntry] = []
        step_entries_noised: List[StepDataEntry] = []

        n_nodes = len(self.nodes)
        indexes = np.arange(n_nodes)
        self.random.shuffle(indexes)

        # percentages to move
        movements = self.random.uniform(self.movement_min, self.movement_max, size=n_nodes // 2)
        if prediction_noise is not None:
            assert len(prediction_noise) == len(movements)
        for i in range(0, n_nodes, 2):
            pair_movement = movements[i // 2]
            if pair_movement > 1:
                pair_movement = 1
            add_node: Node = self.nodes[indexes[i]]
            remove_node: Node = self.nodes[indexes[i + 1]]
            # quantity on which we compute the percentage that we will move
            capacity = remove_node.bikes
            capacity_to_move = np.round(capacity * pair_movement, 0)
            add_node_bikes_new, remove_node_bikes_new = self._get_new_bikes_values(
                add_node, remove_node, capacity_to_move)

            # TODO: StepDataEntry should keep track of the pair and should contains:
            #  the quantity exchanged, the add and remove node and the time step
            step_entries.append(StepDataEntry(add_node_bikes_new, add_node.name, step))
            step_entries.append(StepDataEntry(remove_node_bikes_new, remove_node.name, step))

            if prediction_noise is not None:
                noise = prediction_noise[i // 2]
                capacity_to_move_noised = capacity_to_move + noise

                a_node_bikes_pred, r_node_bikes_pred = self._get_new_bikes_values(
                    add_node, remove_node, capacity_to_move_noised)
                step_entries_noised.append(StepDataEntry(a_node_bikes_pred, add_node.name, step))
                step_entries_noised.append(StepDataEntry(r_node_bikes_pred, remove_node.name, step))

        # TODO: As well as the StepDataEntry, the StepData keeps track of the pair movement
        demand = StepData(*step_entries)
        if prediction_noise is not None:
            demand_noised = StepData(*step_entries_noised)
            self.predicted_demand_clean = demand
            return demand_noised
        else:
            return demand

    def _get_new_bikes_values(
            self,
            add_node: Node,
            remove_node: Node,
            capacity_to_move: float
    ):
        # check if the current capacity + the quantity moved exceeds the number of slots available
        if add_node.empty_slots < capacity_to_move:
            add_node_bikes_new = add_node.total_slots
            remove_node_bikes_new = remove_node.bikes - (add_node.total_slots - add_node.bikes)
        else:
            add_node_bikes_new = add_node.bikes + capacity_to_move
            remove_node_bikes_new = remove_node.bikes - capacity_to_move
        return max(0, add_node_bikes_new), max(0, remove_node_bikes_new)

    def _update_data(self, step: Step) -> StepData:
        if self.last_step_update != step:
            # if predicted data is available, use it (without noise), but check if the step match
            self.last_step_update = step
            if self.use_predictions and self.predicted_demand_clean is not None \
                    and step == self.predicted_demand_clean.step:
                return self.predicted_demand_clean
            else:
                return self._generate_step_data(step, prediction_noise=None)

    def _predict_demand(self, step: Step) -> StepData:
        size = len(self.nodes) // 2
        prediction_noise = np.round(
            self.random.normal(self.gaussian_noise['mean'], self.gaussian_noise['std'], size=size),
            0)
        return self._generate_step_data(step, prediction_noise=prediction_noise)
