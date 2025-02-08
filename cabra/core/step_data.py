import json
from collections import namedtuple
from dataclasses import dataclass
from datetime import datetime
from typing import Union, Dict, List, Any, Optional

import numpy as np

from cabra.common.enum_utils import ExtendedEnum
from cabra.core.timestep import Step

StepDataEntry = namedtuple('StepDataEntry', 'value add_node remove_node step')


class SavingFormat(str, ExtendedEnum):
    FullData = 'full-data'
    OnlyData = 'only-data'
    Array = 'array'


@dataclass()
class NodeStepData:
    started_out_interval: int
    ended: Dict[str, Dict[str, int or str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'started_out_interval': self.started_out_interval,
            'ended': self.ended
        }


@dataclass()
class NodeStepDataCDRC:
    bikes: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'bikes': self.bikes,
        }


@dataclass()
class StepWeather:
    condition: str
    temperature: float
    wind_speed: float


class DataEntryException(Exception):

    def __init__(self, message, *args):
        super(DataEntryException, self).__init__(message, *args)


class StepData:

    def __init__(self, step: Step, date: datetime, weather: Optional[StepWeather] = None):
        """
        Step data contains for each node the number of available bikes for a specific time step.
        """
        self._data: Dict[str, Union[NodeStepData, NodeStepDataCDRC]] = {}
        self.step: Step = step
        self.date: datetime = date
        self.weather: StepWeather = weather
        self._iter_status = 0
        self.data_index = 0

    def populate_data(self, add_node: str, remove_nodes: Dict[str, int or str], out_interval: int):
        """
        We set the data structure relative to add_node.
        Variable remove_nodes contains the id of the nodes that have trips ended on add_node.
        So the ended nodes are the remove nodes from which we take the bikes, and we set them on the add node.
        In out_interval is stored the number of bikes that left add_node station
        but they haven't finished their trip in the current time interval,
        they are the ongoing bikes for add_node station.
        Examples:
            add_node = 'node_1'
            remove_nodes = {'node_2': {'station_id': 'node_2', 'n_bikes': 2}}
            out_interval = 0
            this results in a removal of 2 bikes from node_2 and an add of 2 bikes on node_1
        """
        self._data[add_node] = NodeStepData(out_interval, remove_nodes)

    def populate_cdrc_data(self, node: str, quantity: int):
        self._data[node] = NodeStepDataCDRC(quantity)

    def get_base_station_values(self, base_station):
        return self._data[base_station]

    def __getitem__(self, item):
        return self._data[item]

    def __iter__(self):
        return iter(self._data)

    def items(self):
        return self._data.items()

    def to_array(self) -> List[float]:
        return list(self._data.values())

    def to_numpy(self, *args, **kwargs):
        return np.array(self.to_array(), *args, **kwargs)

    def to_dict(self):
        return {
            'step': self.step.to_dict(),
            'data': {key: n.to_dict() for key, n in self._data.items()}
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    def __str__(self):
        return f'<StepData step={self.step} data={self._data} >'

    def to_saving_format(self, saving_format: SavingFormat):
        if saving_format == SavingFormat.FullData:
            return self.to_dict()
        if saving_format == SavingFormat.OnlyData:
            return self.to_dict()['data']
        if saving_format == SavingFormat.Array:
            return self.to_array()

    @classmethod
    def from_dict(cls, dict_values):
        data = cls()
        data._data = dict_values['data']
        data.step = Step(**dict_values['step'])
        return data

    @classmethod
    def from_json(cls, string_value):
        return StepData.from_dict(json.loads(string_value))
