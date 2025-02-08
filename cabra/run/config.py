import os
from typing import Union, List, Optional

from cabra.__version__ import __version__
from cabra.common.config import AbstractConfig, load_config_dict, set_sub_config, LoggerConfig, \
    ConfigValueError
from cabra.common.data_structure import RunMode
from cabra.common.object_handler import SaverMode
from cabra.emulator.config import EmulatorConfig
from cabra.emulator.data_structure import ModelTypes
from cabra.environment.config import EnvConfig
from cabra.environment.data_structure import NodesConfigType


class RandomSeedsConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.training = 10
        self.evaluation: List[int] = [1000]  # [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        super(RandomSeedsConfig, self).__init__('RandomSeedsConfig', **configs_to_override)


class TensorboardConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.enabled: bool = False
        self.save_path: str = 'tensorboard'
        self.save_model_graph: bool = False
        super(TensorboardConfig, self).__init__('TensorboardConfig', **configs_to_override)

    def get_save_path(self) -> str:
        return os.path.join(self.base_data_dir, self.save_path)


class SaverConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.enabled: bool = True
        self.save_agent: bool = True
        self.mode: SaverMode = SaverMode.Disk
        self.base_path = 'results'
        self.default_bucket = 'bikesharing'
        self.save_prefix: str = ''
        self.save_name_with_uuid: bool = True
        self.save_name_with_date: bool = True
        self.save_name: str = ''
        self.stats_condensed: bool = True
        self.checkpoint_frequency: int = 1000000
        self.tensorboard: TensorboardConfig = set_sub_config('tensorboard', TensorboardConfig, **configs_to_override)
        super(SaverConfig, self).__init__('SaverConfig', **configs_to_override)

    def get_base_path(self) -> str:
        return os.path.join(self.base_data_dir, self.base_path)

    def _after_override_configs(self):
        if isinstance(self.mode, str):
            try:
                self.mode: SaverMode = SaverMode(self.mode)
            except Exception:
                raise ConfigValueError('mode', self.mode, module=self.name(),
                                       extra_msg=f'Possible values are: {SaverMode.list()}')


class RunConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        #  Filter out also the unnecessary config
        self.run_mode: RunMode = RunMode.Train
        self.training_steps: int = 400
        self.evaluation_episodes: int = 1
        self.debug_frequency: int = 1
        self.info_frequency: int = 1
        self.evaluation_frequency: int = 5
        self.use_best_evaluation: bool = True
        self.evaluation_steps: int = -1
        self.keep_metric: str = 'evaluation/avg/reward'
        self.metrics_window: int = 5
        self.eval_pool_size: int = 0  # zero to disable async evaluation
        super(RunConfig, self).__init__('RunConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.run_mode, str):
            try:
                self.run_mode: RunMode = RunMode(self.run_mode)
            except Exception:
                raise ConfigValueError('run_mode', self.run_mode, module=self.name(),
                                       extra_msg=f'Possible values are: {RunMode.list()}')


class RedisConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.enabled: bool = True
        super(RedisConfig, self).__init__('RedisConfig', **configs_to_override)


class MultiRunParamConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.key: str = 'environment.agent.double_dqn.learning_rate'
        self.key_short: str = 'lr'
        self.value: Union[str, int, float, list, dict] = 0.1
        self.filename_key_val: str = 'lr=0.1'
        super(MultiRunParamConfig, self).__init__(MultiRunParamConfig, **configs_to_override)


class MultiRunConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.is_multi_run: bool = False
        self.multi_run_code: Optional[str] = None
        self.multi_run_params: List[MultiRunParamConfig] = [
            # MultiRunParamConfig(
            #     key='environment.agent.double_dqn.learning_rate',
            #     value=0.1,
            #     filename_key_val='lr=0.1'
            # ),
            # MultiRunParamConfig(
            #     key='environment.agent.double_dqn.target_net_update_frequency',
            #     value=3600,
            #     filename_key_val='target-update=3600'
            # )
        ]
        super(MultiRunConfig, self).__init__(MultiRunConfig, **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.multi_run_params, list):
            multi_run_params: List[MultiRunParamConfig] = []
            for param in self.multi_run_params:
                if isinstance(param, MultiRunParamConfig):
                    multi_run_params.append(param)
                elif isinstance(param, dict):
                    multi_run_params.append(MultiRunParamConfig(**param))
                else:
                    raise ConfigValueError('multi_run_params', self.multi_run_params, self.name(),
                                           extra_msg='multi_run_params entries must be an object')
            self.multi_run_params: List[MultiRunParamConfig] = multi_run_params


class SingleRunConfig(AbstractConfig):

    def __init__(self, root_dir, **configs_to_override):
        self.version = __version__
        self.run: RunConfig = set_sub_config('run', RunConfig, **configs_to_override)
        self.saver: SaverConfig = set_sub_config('saver', SaverConfig, **configs_to_override)
        self.emulator: EmulatorConfig = set_sub_config('emulator', EmulatorConfig, root_dir, **configs_to_override)
        self.environment: EnvConfig = set_sub_config('environment', EnvConfig, root_dir, **configs_to_override)
        self.logger: LoggerConfig = set_sub_config('logger', LoggerConfig, 'global', **configs_to_override)
        self.random_seeds: RandomSeedsConfig = set_sub_config('random_seeds', RandomSeedsConfig, **configs_to_override)
        self.redis: RedisConfig = set_sub_config('redis', RedisConfig, **configs_to_override)
        self.multi_run: MultiRunConfig = set_sub_config('multi_run', MultiRunConfig, **configs_to_override)
        super(SingleRunConfig, self).__init__(root_dir=root_dir,
                                              config_object_name='SingleRunConfig', **configs_to_override)

    def _after_override_configs(self):
        if self.environment.nodes.use_predictions is True:
            self.emulator.model.use_predictions = True

        if self.emulator.model.type == ModelTypes.TestData:
            nodes_path = self.emulator.model.test_data_model.test_nodes.nodes_path
            n_nodes = self.emulator.model.test_data_model.test_nodes.n_nodes
            self.environment.nodes.n_nodes = n_nodes
            self.environment.nodes.nodes_config = NodesConfigType.Loaded
            self.environment.nodes.loaded_config.nodes_load_path = nodes_path
            self.environment.nodes.loaded_config.bikes_percentage_from_data = True


def get_single_run_config(root_dir, config_path: str = None, log_level: int or None = None) -> SingleRunConfig:
    config = SingleRunConfig(root_dir)
    if config_path is not None:
        config_dict = load_config_dict(config_path, root_dir)
        config.set_configs(**config_dict)
    if log_level is not None:
        config.set_configs(logger={'level': log_level})
    return config
