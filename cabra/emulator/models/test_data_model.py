import json
import math
from datetime import datetime, timedelta
from logging import Logger
from typing import Dict, List, Optional

from numpy.random import RandomState

from cabra import SingleRunConfig, logger
from cabra.common.data_structure import RunMode, Done
from cabra.common.filesystem import create_directory_from_filepath
from cabra.core.step import Step
from cabra.core.step_data import StepData
from cabra.emulator import EmulatorConfig
from cabra.emulator.data_structure import ModelTypes
from cabra.emulator.models import AbstractModel
from cabra.environment.node import Node


class TestDataModel(AbstractModel):

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
        super().__init__(
            nodes, model_name=ModelTypes.TestData,
            random_state=random_state, random_seed=random_seed,
            emulator_configuration=emulator_configuration, log=log, disable_log=disable_log, run_mode=run_mode)
        self.n_nodes: int = self.emulator_config.model.test_data_model.test_nodes.n_nodes
        self.nodes_in_shortage: int = self.emulator_config.model.test_data_model.nodes_in_shortage
        assert self.nodes_in_shortage <= self.n_nodes // 2
        self.update_frequency: Step = self.emulator_config.model.test_data_model.update_frequency
        self.shuffle_frequency: Optional[Step] = self.emulator_config.model.test_data_model.shuffle_frequency
        self.min_date = datetime.fromisoformat(self.emulator_config.model.test_data_model.min_date)
        self.max_date = datetime.fromisoformat(self.emulator_config.model.test_data_model.max_date)
        self.episode_length: Step = self.emulator_config.model.test_data_model.episode_length

        self.dataset: Dict[str, dict] = {}
        self.current_date = self.min_date
        self.next_done_date = self.min_date + timedelta(seconds=self.episode_length.total_steps)
        self.current_index: int = 0

        self.last_demand: Optional[StepData] = None

    def _update_data(self, step: Step) -> StepData:
        index_updated = False
        if step.total_steps > 0 and step.total_steps % self.update_frequency.total_steps == 0:
            # we update the index, we load the new demand, and we update the next_change_step
            self.current_index += 1
            self.current_date += timedelta(seconds=self.update_frequency.total_steps)
            index_updated = True
        # we use the current index for loading the current demand, and we return it
        return self._get_step_data(step, index_updated)

    def _get_step_data(self, step: Step, index_updated: bool) -> StepData:
        if not index_updated and self.last_demand:
            return self.last_demand
        else:
            demand = StepData(step, self.current_date)
            is_shuffle_step = self._is_shuffle_step(step)
            other_nodes: Dict[Node, int] = {}
            next_shortage_nodes = []
            if is_shuffle_step:
                node_indexes_shuffled = self.random.choice(
                    [n.index for n in self.nodes], self.nodes_in_shortage, replace=False).tolist()
                for node in self.nodes:
                    if node.index in node_indexes_shuffled:
                        next_shortage_nodes.append(node)
                    else:
                        other_nodes[node] = node.bikes

            should_stop = False
            for add_node in self.nodes:
                if is_shuffle_step:
                    # do the shuffle
                    if add_node in next_shortage_nodes and not should_stop:
                        bikes_for_shortage = add_node.empty_slots
                        ended_dict = {}
                        for r_node, r_node_bikes in other_nodes.items():
                            if r_node_bikes >= bikes_for_shortage:
                                ended_dict[r_node.name] = {
                                    'station_id': r_node.name, 'n_bikes': bikes_for_shortage}
                                other_nodes[r_node] -= bikes_for_shortage
                                break
                            else:
                                ended_dict[r_node.name] = {
                                    'station_id': r_node.name, 'n_bikes': r_node_bikes}
                                bikes_for_shortage -= r_node_bikes
                                other_nodes[r_node] = 0
                                # if all next_empty_nodes are empty we should stop
                        all_empty = True
                        for node, bikes in other_nodes.items():
                            if bikes > node.shortage_threshold:
                                all_empty = False
                        should_stop = all_empty is True
                        demand.populate_data(add_node.name, ended_dict, 0)
                    else:
                        demand.populate_data(add_node.name, {}, 0)
                else:
                    # do an empty step
                    demand.populate_data(add_node.name, {}, 0)
            demand.data_index = self.current_index
            self.last_demand = demand
            return demand

    def _is_shuffle_step(self, step: Step) -> bool:
        if self.shuffle_frequency is None:
            return self.current_index == 0
        else:
            return step.total_steps % self.shuffle_frequency.total_steps == 0

    def _predict_demand(self, step: Step) -> StepData:
        raise NotImplementedError('predict demand is not implemented yet')

    def is_done(self, step: Step) -> Done:
        done = Done.NotDone
        if self.current_date >= self.next_done_date or self.current_date >= self.max_date:
            done = Done.VirtualDone if self.current_date < self.max_date else Done.Done
            self.next_done_date += timedelta(seconds=self.episode_length.total_steps)
        return done

    def reset(self, reset_nodes=True):
        super().reset(reset_nodes)
        if self.next_done_date >= self.max_date:
            # it is a full reset
            self.current_date = self.min_date
            self.current_index: int = 0
            self.next_done_date = self.min_date + timedelta(seconds=self.episode_length.total_steps)


def create_nodes_file(
        config: SingleRunConfig
):
    random: RandomState = RandomState(seed=config.random_seeds.training)
    nodes_config = config.emulator.model.test_data_model.test_nodes
    logger.info(f'Generating {nodes_config.n_nodes} test nodes in {nodes_config.nodes_path}')
    cols = math.ceil(math.sqrt(nodes_config.n_nodes))
    nodes: Dict[str, dict] = {}
    nodes_ids: List[str] = [f'node_{index + 1}' for index in range(nodes_config.n_nodes)]

    distances: Dict[str, Dict[str, float]] = {}

    rows = math.ceil(nodes_config.n_nodes / cols)
    index = 0
    for row in range(rows):
        for col in range(cols):
            if index < nodes_config.n_nodes:
                node_id = f'node_{index + 1}'
                capacity = int(random.normal(nodes_config.nodes_capacity_avg, nodes_config.nodes_capacity_std))
                nodes[node_id] = {
                    'name': node_id,
                    'index': index,
                    'position': {'type': 'FakePoint', 'coordinates': [col, row]},
                    'capacity': capacity,
                    'distances': get_node_distances(node_id, nodes_ids, distances, random, config),
                    'bikes_percentage': nodes_config.nodes_bikes_percentage
                }
                index += 1
    nodes_data = {
        'ids': nodes_ids,
        'nodes': nodes
    }
    create_directory_from_filepath(nodes_config.get_nodes_path())
    with open(nodes_config.get_nodes_path(), 'w') as f:
        json.dump(nodes_data, f, indent=2)
    logger.info('Nodes generated')


def get_node_distances(
        node_id: str,
        nodes_ids: List[str],
        distances: Dict[str, Dict[str, float]],
        random: RandomState,
        config: SingleRunConfig
) -> Dict[str, float]:
    dist_avg = config.emulator.model.test_data_model.test_nodes.nodes_distance_avg
    dist_std = config.emulator.model.test_data_model.test_nodes.nodes_distance_std
    node_distances: Dict[str, float] = {node_id: 0.0}
    for n_id in nodes_ids:
        if n_id != node_id:
            if n_id not in distances or n_id not in distances[n_id]:
                node_distances[n_id] = random.normal(dist_avg, dist_std)
            else:
                node_distances[n_id] = distances[n_id][node_id]
    for n_id, n_dist in node_distances.items():
        if n_id != node_id:
            if n_id not in distances:
                distances[n_id] = {node_id: n_dist}
            else:
                if node_id not in distances[n_id]:
                    distances[n_id][node_id] = n_dist
        if node_id not in distances:
            distances[node_id] = {node_id: 0.0}
        else:
            if n_id not in distances[node_id]:
                distances[node_id][n_id] = n_dist
    return node_distances
