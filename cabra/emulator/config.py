import os
import sys

from cabra.emulator.data_structure import ModelTypes

if sys.version_info >= (3, 8):
    from typing import Dict, List, Union, TypedDict, Tuple, Optional
else:
    from typing import Dict, List, Union, Tuple
    from typing_extensions import TypedDict

from cabra.common.config import AbstractConfig, ConfigValueError, load_config_dict, set_sub_config, \
    LoggerConfig
from cabra.common.enum_utils import ExtendedEnum
from cabra.common.object_handler import SaverMode
from cabra.common.filesystem import ROOT_DIR
from cabra.core.step import Step
from cabra.core.step_data import SavingFormat


class ChangeModelDistribution(str, ExtendedEnum):
    Random = 'random'
    RoundRobin = 'round-robin'


class RealisticInterval(TypedDict):
    goal: Tuple[float, float]
    reach_interval: Step
    stay_interval: Step


class NodeProfile(TypedDict):
    absolute_value: float
    weekend_multiplier: float
    stress_multiplier: float
    stress_probability: float
    std: float
    values: List[RealisticInterval]


class ModelConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.type: ModelTypes = ModelTypes.CDRCData
        # self.model_disk: ModelDiskConfig = set_sub_config('model_disk', ModelDiskConfig, **configs_to_override)
        self.synthetic_model: SyntheticModelConfig = set_sub_config('synthetic_model',
                                                                    SyntheticModelConfig, **configs_to_override)
        self.realdata_model: RealDataModelConfig = set_sub_config('realdata_model',
                                                                  RealDataModelConfig, **configs_to_override)
        self.cdrc_data_model: CDRCDataModelConfig = set_sub_config('cdrc_data_model',
                                                                   CDRCDataModelConfig, **configs_to_override)
        self.test_data_model: TestDataModelConfig = set_sub_config('test_data_model',
                                                                   TestDataModelConfig, **configs_to_override)
        self.use_predictions: bool = False
        super(ModelConfig, self).__init__('ModelConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.type, str):
            try:
                self.type: ModelTypes = ModelTypes(self.type)
            except Exception:
                raise ConfigValueError('loader_mode', self.type, module=self.name(),
                                       extra_msg=f'Possible values are: {ModelTypes.list()}')


# class ModelDiskConfig(AbstractConfig):
#
#     def __init__(self, **configs_to_override):
#         self.use_disk: bool = False
#         self.episodes_to_generate: Dict[str, int] = {
#             'training': 100,
#             'validation': 1,
#             'evaluation': 1
#         }
#         self.seeds: Dict[str, List[int]] = {
#             'training': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
#             'validation': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
#             'evaluation': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
#         }
#         self.base_folder: str = os.path.join(ROOT_DIR, 'data', 'models')
#         super(ModelDiskConfig, self).__init__('ModelDiskConfig', **configs_to_override)


class SyntheticModelConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.movement_range: Tuple[float, float] = (0.3, 1.1)
        self.gaussian_noise: Dict[str, float] = {
            'mean': 0,
            'std': 2.5
        }
        super(SyntheticModelConfig, self).__init__('SyntheticModelConfig', **configs_to_override)


class RealDataModelConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.train_dataset_path: str = 'datasets/citibike-zones/5-zones_nodes-training/dataset.json'
        self.eval_dataset_path: str = 'datasets/citibike-zones/5-zones_nodes-evaluation/dataset.json'
        self.nodes_path: str = 'datasets/citibike-zones/5-zones_nodes-training/nodes.json'
        self.weather_info_path: str = 'datasets/citibike-zones/weather_info.json'
        self.update_frequency: Step = Step.from_str('10m')
        self.episode_length: Step = Step.from_str('1W')
        super(RealDataModelConfig, self).__init__('RealDataModelConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.update_frequency, str):
            self.update_frequency = Step.from_str(self.update_frequency)
        if isinstance(self.episode_length, str):
            self.episode_length = Step.from_str(self.episode_length)

    def get_training_dataset_path(self) -> str:
        return os.path.join(self.base_data_dir, self.train_dataset_path)

    def get_evaluation_dataset_path(self) -> str:
        return os.path.join(self.base_data_dir, self.eval_dataset_path)

    def get_nodes_path(self) -> str:
        return os.path.join(self.base_data_dir, self.nodes_path)

    def get_weather_info_path(self) -> str:
        return os.path.join(self.base_data_dir, self.weather_info_path)


class CDRCDataModelConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.train_dataset_path: str = 'datasets/cdrc/dublin/dataset-training/dataset.json'
        self.eval_dataset_path: str = 'datasets/cdrc/dublin/dataset-evaluation/dataset.json'
        self.nodes_path: str = 'datasets/cdrc/dublin/dataset-training/nodes.json'
        # self.weather_info_path: str = 'datasets/citibike-zones/weather_info.json'
        self.update_frequency: Step = Step.from_str('10m')
        self.episode_length: Step = Step.from_str('1W')
        self.eval_every_week: bool = False
        self.store_demand_stats: bool = False
        super(CDRCDataModelConfig, self).__init__('CDRCDataModelConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.update_frequency, str):
            self.update_frequency = Step.from_str(self.update_frequency)
        if isinstance(self.episode_length, str):
            self.episode_length = Step.from_str(self.episode_length)

    def get_training_dataset_path(self) -> str:
        return os.path.join(self.base_data_dir, self.train_dataset_path)

    def get_evaluation_dataset_path(self) -> str:
        return os.path.join(self.base_data_dir, self.eval_dataset_path)

    def get_nodes_path(self) -> str:
        return os.path.join(self.base_data_dir, self.nodes_path)

    # def get_weather_info_path(self) -> str:
    #     return os.path.join(self.base_data_dir, self.weather_info_path)


class TestNodeConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.n_nodes: int = 4
        self.nodes_path: str = f'datasets/test_data/{self.n_nodes}_nodes/nodes.json'
        self.nodes_capacity_avg: float = 60
        self.nodes_capacity_std: float = 6
        self.nodes_distance_avg: float = 500
        self.nodes_distance_std: float = 50
        self.nodes_bikes_percentage: float = 0.6
        super().__init__('TestNodeConfig', **configs_to_override)

    def get_nodes_path(self) -> str:
        return os.path.join(self.base_data_dir, self.nodes_path)


class TestDataModelConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.test_nodes: TestNodeConfig = set_sub_config('test_nodes', TestNodeConfig, **configs_to_override)
        self.nodes_in_shortage: int = 2
        self.min_date: str = '2022-01-01'
        self.max_date: str = '2022-02-01'
        self.update_frequency: Step = Step.from_str('10m')
        self.shuffle_frequency: Optional[Step] = Step.from_str('1w')
        self.episode_length: Step = Step.from_str('1W')
        super().__init__('TestDataModelConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.update_frequency, str):
            self.update_frequency = Step.from_str(self.update_frequency)
        if isinstance(self.shuffle_frequency, str):
            self.shuffle_frequency = Step.from_str(self.shuffle_frequency)
        if isinstance(self.episode_length, str):
            self.episode_length = Step.from_str(self.episode_length)


class EmulatorConfig(AbstractConfig):

    def __init__(self, root_dir, **configs_to_override):
        self.model: ModelConfig = set_sub_config('model', ModelConfig, **configs_to_override)
        # self.saver: SaverConfig = set_sub_config('saver', SaverConfig, **configs_to_override)
        # self.simulation: SimulationConfig = set_sub_config('simulation', SimulationConfig, **configs_to_override)
        self.logger: LoggerConfig = set_sub_config('logger', LoggerConfig, 'emulator', **configs_to_override)
        super(EmulatorConfig, self).__init__(config_object_name='EmulatorConfig',
                                             root_dir=root_dir, **configs_to_override)
        self.export_exclude.append('logger')


def get_emulator_config(root_dir, config_path: str = None, log_level: int or None = None) -> EmulatorConfig:
    config = EmulatorConfig(root_dir)
    if config_path is not None:
        config_dict = load_config_dict(config_path, root_dir)
        config.set_configs(**config_dict)
    if log_level is not None:
        config.set_configs(logger={'level': log_level})
    return config
