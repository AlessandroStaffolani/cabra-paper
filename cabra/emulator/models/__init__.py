from logging import Logger
from typing import Optional, List

from numpy.random import RandomState

from cabra.emulator import EmulatorConfig
from cabra.emulator.data_structure import ModelTypes
from cabra.emulator.models.abstract import AbstractModel
from cabra.emulator.models.cdrc_data_model import CDRCDataModel
from cabra.emulator.models.real_data_model import RealDataModel
from cabra.emulator.models.synthetic_model import SyntheticModel
from cabra.emulator.models.test_data_model import TestDataModel
from cabra.common.data_structure import RunMode
from cabra.environment.node import Node

MAPPINGS = {
    ModelTypes.RealData: RealDataModel,
    ModelTypes.CDRCData: CDRCDataModel,
    ModelTypes.SyntheticDataGenerator: SyntheticModel,
    ModelTypes.TestData: TestDataModel
}


def get_value(key, kwargs, default):
    if key in kwargs:
        return kwargs[key]
    else:
        return default


def create_model_from_type(
        model_type: ModelTypes,
        nodes: List[Node],
        em_config: EmulatorConfig,
        random_state: RandomState,
        log: Optional[Logger] = None,
        disable_log: bool = False,
        run_mode: RunMode = RunMode.Train,
        random_seed: int = 42,
        **kwargs) -> AbstractModel:
    if model_type in MAPPINGS:
        model_class = MAPPINGS[model_type]
        return model_class(
            nodes,
            random_state=random_state,
            random_seed=random_seed,
            emulator_configuration=em_config,
            log=log,
            disable_log=disable_log,
            run_mode=run_mode,
            **kwargs,
        )
    else:
        raise AttributeError(f'ModelTypes {model_type} is not available')
