import json
import os.path
import sys

from cabra.core.step import Step
from cabra.environment.agent.learning_scheduler import LRScheduler

if sys.version_info >= (3, 8):
    from typing import Union, List, Dict, Any, Optional
else:
    from typing import Union, List, Dict, Any, Optional

from cabra.common.config import AbstractConfig, ConfigValueError, load_config_dict, \
    LoggerConfig, set_sub_config
from cabra.common.distance_helper import DistanceMode
from cabra.common.object_handler import SaverMode
from cabra.environment.agent import AgentType
from cabra.environment.data_structure import StateFeatureName, RewardFunctionType, \
    NodesConfigType, DiscretizeMode


def get_net_layers(
        config: AbstractConfig,
        prop: str,
        layer_config_class,
        override_activation_func: bool,
        activation_func: str,
        override_batch_normalization: bool,
        add_batch_normalization: bool
) -> List['CNNLayerConfig'] or List['FullyConnectedLayerConfig']:
    layers = config[prop]
    if len(layers) > 0:
        layers_casted = []
        for i, layer in enumerate(layers):
            if isinstance(layer, dict):
                layers_casted.append(layer_config_class(**layer))
            elif isinstance(layer, layer_config_class):
                layers_casted.append(layer)
            else:
                raise AttributeError(f'layer type invalid for prop {prop} in {config.name()}')
            if override_activation_func:
                layers_casted[i].activation = activation_func
            if override_batch_normalization:
                layers_casted[i].add_batch_normalization = add_batch_normalization
        return layers_casted
    else:
        return []


class NodesGeneratedConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.total_slots_avg: int = 40
        self.total_slots_std: int = 5
        self.bikes_percentage: float = 0.6
        super(NodesGeneratedConfig, self).__init__('NodesGeneratedConfig', **configs_to_override)


class NodesLoadedConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.nodes_load_path: str = 'datasets/cdrc/dublin/dataset-training/nodes.json'
        self.zones_load_path: str = 'datasets/cdrc/dublin/dataset-training/zones.json'
        self.bikes_percentage_from_data: bool = True
        self.bikes_percentage: float = 0.45
        self.normalize_coordinates: bool = True
        super(NodesLoadedConfig, self).__init__('NodesLoadedConfig', **configs_to_override)

    def get_nodes_load_path(self) -> str:
        return os.path.join(self.base_data_dir, self.nodes_load_path)

    def get_zones_load_path(self) -> str:
        return os.path.join(self.base_data_dir, self.zones_load_path)


class NodesConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.n_nodes: int = 4
        self.shortage_threshold: int = 0
        self.distance_mode: DistanceMode = DistanceMode.L1
        self.nodes_config: NodesConfigType = NodesConfigType.Loaded
        self.use_predictions: bool = False
        self.nodes_features: List[str] = [
            # 'shortage_flag',
            'bikes',
            # 'empty_slots',
            # 'ongoing_trips',
            # 'bikes_critic_flag',
            # 'empty_critic_flag',
            # 'critical_flag',
            'position',
        ]
        self.critical_threshold: float = 0.2
        self.critical_normalized: bool = True
        self.generated_config: NodesGeneratedConfig = set_sub_config('generated_config',
                                                                     NodesGeneratedConfig, **configs_to_override)
        self.loaded_config: NodesLoadedConfig = set_sub_config('loaded_config',
                                                               NodesLoadedConfig, **configs_to_override)
        super(NodesConfig, self).__init__('NodesConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.distance_mode, str):
            try:
                self.distance_mode: DistanceMode = DistanceMode(self.distance_mode)
            except Exception:
                raise ConfigValueError('distance_mode', self.distance_mode, module=self.name(),
                                       extra_msg=f'Possible values are: {DistanceMode.list()}')
        if isinstance(self.nodes_config, str):
            try:
                self.nodes_config: NodesConfigType = NodesConfigType(self.nodes_config)
            except Exception:
                raise ConfigValueError('nodes_config', self.nodes_config, module=self.name(),
                                       extra_msg=f'Possible values are: {NodesConfigType.list()}')
        if self.nodes_config == NodesConfigType.Loaded:
            with open(self.loaded_config.get_nodes_load_path(), 'r') as f:
                nodes = json.load(f)
            self.n_nodes = len(nodes['ids'])


class TrucksConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.n_trucks: int = 3
        self.initial_node_index: int = 0
        self.initial_load_level: int = 0
        self.capacity: int = 20
        self.move_speed_avg: float = 18 / 3.6  # in meter/second, so we divide 40 k/h by 3.6
        self.move_speed_std: float = 0.8
        self.reposition_time_avg: float = 60  # seconds for a single reposition
        self.reposition_time_std: float = 0.5
        self.truck_features: List[str] = [
            'position',
            'load',
            'busy',
        ]
        super(TrucksConfig, self).__init__('TrucksConfig', **configs_to_override)


class StateConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.common_features: List[StateFeatureName] = [
            # StateFeatureName.PreviousAction,
            # StateFeatureName.SelectedZone,
            StateFeatureName.CurrentZone,
            StateFeatureName.CurrentTruckFull,
            StateFeatureName.TruckFeatures,
            StateFeatureName.DatasetTime,
            # StateFeatureName.Weather,
        ]
        self.target_features: List[StateFeatureName] = []
        self.quantity_features: List[StateFeatureName] = []
        self.zone_features: List[StateFeatureName] = [
            # StateFeatureName.PreviousFullAction,
            StateFeatureName.Zones,
            StateFeatureName.TruckFeatures,
            StateFeatureName.CurrentTruck,
            StateFeatureName.DatasetTime,
            # StateFeatureName.Weather,
        ]
        self.normalized: bool = True
        self.additional_properties: Dict[str, Any] = {
            'units_to_skip': ['second_step', 'second', 'minute', 'week', 'year', 'total_steps'],
        }
        self.stack_n_states: int = 1
        self.weather_info_path: str = 'datasets/citibike/weather_info.json'
        super(StateConfig, self).__init__('StateConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.common_features, list) and isinstance(self.common_features[0], str):
            self.common_features: List[StateFeatureName] = [
                StateFeatureName(feature) for feature in self.common_features]
        else:
            ConfigValueError('common_features', self.common_features, module=self.name(),
                             extra_msg=f'Possible values are: {StateFeatureName.list()}')
        if isinstance(self.target_features, list) and len(self.target_features) > 0 and isinstance(
                self.target_features[0], str):
            self.target_features: List[StateFeatureName] = [
                StateFeatureName(feature) for feature in self.target_features]
        else:
            ConfigValueError('target_features', self.target_features, module=self.name(),
                             extra_msg=f'Possible values are: {StateFeatureName.list()}')
        if isinstance(self.quantity_features, list) and len(self.quantity_features) > 0 and isinstance(
                self.quantity_features[0], str):
            self.quantity_features: List[StateFeatureName] = [
                StateFeatureName(feature) for feature in self.quantity_features]
        else:
            ConfigValueError('quantity_features', self.quantity_features, module=self.name(),
                             extra_msg=f'Possible values are: {StateFeatureName.list()}')
        if isinstance(self.zone_features, list) and len(self.zone_features) > 0 and isinstance(
                self.zone_features[0], str):
            self.zone_features: List[StateFeatureName] = [
                StateFeatureName(feature) for feature in self.zone_features]
        else:
            ConfigValueError('zone_features', self.zone_features, module=self.name(),
                             extra_msg=f'Possible values are: {StateFeatureName.list()}')

    def get_weather_info_path(self) -> str:
        return os.path.join(self.base_data_dir, self.weather_info_path)


class ActionSpaceConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.quantity_movements: List[int] = [i for i in range(20)]
        self.min_quantity: int = 0
        self.use_continuous_action_space: bool = False
        self.discretize_mode: DiscretizeMode = DiscretizeMode.Greedy
        self.add_wait_space: bool = False
        super(ActionSpaceConfig, self).__init__('ActionSpaceConfig', **configs_to_override)


class RewardConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.type: RewardFunctionType = RewardFunctionType.GlobalShortageAndCost
        self.invalid_action_penalty: int = -100
        self.gamma: float = 1.02
        self.disable_cost: bool = False
        self.training_scaler: float = 1
        self.solved_bonus: float = 1
        self.parameters: Dict[str, Any] = {
            'normalize_cost': True,
            'shortage_weight': 0.5,
            'environment_shortage_weight': 1,
            'cost_weight': 0.5
        }
        super(RewardConfig, self).__init__('RewardConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.type, str):
            try:
                self.type: RewardFunctionType = RewardFunctionType(self.type)
            except Exception:
                raise ConfigValueError('type', self.type, module=self.name(),
                                       extra_msg=f'Possible values are: {RewardFunctionType.list()}')


class PPOConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.lr: float = 0.000003
        self.lr_scheduler: LRScheduler = LRScheduler.Constant
        self.n_steps: int = 8192
        self.batch_size: int = 128
        self.n_epochs: int = 10
        self.gamma: float = 0.9
        self.gae_lambda: float = 0.95
        self.clip_range: float = 0.2
        self.clip_range_vf: Optional[float] = None
        self.normalize_advantage: bool = True
        self.ent_coef: float = 0.01
        self.vf_coef: float = 0.5
        self.max_grad_norm: float = 0.5
        self.shared_net_cnn: List[CNNLayerConfig] = []
        self.shared_net_fcl: List[FullyConnectedLayerConfig] = [
            FullyConnectedLayerConfig(units=128, add_batch_normalization=True, activation='tanh'),
            FullyConnectedLayerConfig(units=128, add_batch_normalization=True, activation='tanh')
        ]
        self.policy_layers: List[FullyConnectedLayerConfig] = [
            FullyConnectedLayerConfig(units=64, add_batch_normalization=True, activation='tanh'),
        ]
        self.value_layers: List[FullyConnectedLayerConfig] = [
            FullyConnectedLayerConfig(units=64, add_batch_normalization=True, activation='tanh'),
        ]
        self.target_kl: Optional[float] = None
        self.log_std_init: float = 0
        self.override_activation_func: bool = False
        self.override_batch_normalization: bool = False
        self.activation_func: str = 'relu'
        self.add_batch_normalization: bool = False
        self.prevent_penalties: bool = True
        self.train_steps_with_constraints: int = 0
        self.deterministic_eval: bool = True
        self.force_constraints_swap: bool = False
        self.train_frequency: int = 1
        super(PPOConfig, self).__init__('PPOConfig', **configs_to_override)

    def _after_override_configs(self):
        self.set_enum_from_string('lr_scheduler', LRScheduler)
        self.shared_net_fcl = get_net_layers(self, 'shared_net_fcl', FullyConnectedLayerConfig,
                                             self.override_activation_func, self.activation_func,
                                             self.override_batch_normalization, self.add_batch_normalization)
        self.shared_net_cnn = get_net_layers(self, 'shared_net_cnn', CNNLayerConfig, self.override_activation_func,
                                             self.activation_func, self.override_batch_normalization,
                                             self.add_batch_normalization)
        self.policy_layers = get_net_layers(self, 'policy_layers', FullyConnectedLayerConfig,
                                            self.override_activation_func, self.activation_func,
                                            self.override_batch_normalization, self.add_batch_normalization)
        self.value_layers = get_net_layers(self, 'value_layers', FullyConnectedLayerConfig,
                                           self.override_activation_func, self.activation_func,
                                           self.override_batch_normalization, self.add_batch_normalization)


class CNNLayerConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.out_channels: int = 32
        self.kernel_size: int = 3
        self.stride: int = 2
        self.padding: int = 2
        self.add_batch_normalization: bool = False
        self.activation: str = 'relu'
        super(CNNLayerConfig, self).__init__('CNNLayerConfig', **configs_to_override)


class FullyConnectedLayerConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.units: int = 64
        self.add_batch_normalization: bool = False
        self.activation: str = 'relu'
        super(FullyConnectedLayerConfig, self).__init__('FullyConnectedLayerConfig', **configs_to_override)


class ConstrainedSpaceConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.critical_threshold: float = 0.2
        self.max_distance: float = 10000
        self.zone_max_distance: float = 10000
        self.zones_filtered_size: int = 3
        super(ConstrainedSpaceConfig, self).__init__('ConstrainedSpaceConfig', **configs_to_override)


class NStepsOracleConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.look_ahead_steps: int = 1
        super().__init__('NStepsOracleConfig', **configs_to_override)


class SimulatedAnnealingConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.max_iterations: int = 1000
        self.cooling_rate: float = 0.95
        super().__init__('SimulatedAnnealingConfig', **configs_to_override)


class ModelLoadConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.path: Optional[str] = None
        self.mode: Optional[SaverMode] = SaverMode.Disk
        self.base_path: str = ''
        self.use_ssh_tunnel: bool = False
        super(ModelLoadConfig, self).__init__('ModelLoadConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.mode, str):
            try:
                self.mode: SaverMode = SaverMode(self.mode)
            except Exception:
                raise ConfigValueError('mode', self.mode, module=self.name(),
                                       extra_msg=f'Possible values are: {SaverMode.list()}')


class ModelLoadWrapperConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.load_model: bool = False
        self.load_model_config: ModelLoadConfig = set_sub_config('load_model_config', ModelLoadConfig,
                                                                 **configs_to_override)
        super(ModelLoadWrapperConfig, self).__init__('ModelLoadWrapperConfig', **configs_to_override)


class AgentConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.type: AgentType = AgentType.ConstrainedPPO
        self.global_parameters: Dict[str, object] = {}
        self.ppo: PPOConfig = set_sub_config('ppo', PPOConfig, **configs_to_override)
        self.n_steps_oracle: NStepsOracleConfig = set_sub_config('n_steps_oracle', NStepsOracleConfig,
                                                                 **configs_to_override)
        self.simulated_annealing: SimulatedAnnealingConfig = set_sub_config(
            'simulated_annealing', SimulatedAnnealingConfig, **configs_to_override)
        self.model_load: ModelLoadWrapperConfig = set_sub_config('model_load', ModelLoadWrapperConfig,
                                                                 **configs_to_override)
        super(AgentConfig, self).__init__('AgentConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.type, str):
            try:
                self.type: AgentType = AgentType(self.type)
            except Exception:
                raise ConfigValueError('type', self.type, module=self.name(),
                                       extra_msg=f'Possible values are: {AgentType.list()}')


class TimeStepConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.stop_step: int = -1
        self.step_per_second: int = 1
        self.step_size: int = 60 * 10  # 10 minutes
        self.stop_date: Union[Step, None] = None
        self.initial_date: Union[Step, None] = None
        super(TimeStepConfig, self).__init__('TimeStepConfig', **configs_to_override)


class ZonesAgentConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.type: AgentType = AgentType.ConstrainedPPO
        self.ppo: PPOConfig = set_sub_config('ppo', PPOConfig, **configs_to_override)
        self.model_load: ModelLoadWrapperConfig = set_sub_config('model_load', ModelLoadWrapperConfig,
                                                                 **configs_to_override)
        super().__init__('ZonesAgentConfig', **configs_to_override)

    def _after_override_configs(self):
        self.set_enum_from_string('type', AgentType)


class ZonesConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.enabled: bool = True
        self.action_space_can_wait: bool = True
        self.agent: ZonesAgentConfig = set_sub_config('agent', ZonesAgentConfig, **configs_to_override)
        super().__init__('ZonesConfig', **configs_to_override)


class EnvConfig(AbstractConfig):

    def __init__(self, root_dir, **configs_to_override):
        self.nodes: NodesConfig = set_sub_config('nodes', NodesConfig, **configs_to_override)
        self.trucks: TrucksConfig = set_sub_config('trucks', TrucksConfig, **configs_to_override)
        self.agent: AgentConfig = set_sub_config('agent', AgentConfig, **configs_to_override)
        self.constrained_space: ConstrainedSpaceConfig = set_sub_config('constrained_space', ConstrainedSpaceConfig,
                                                                        **configs_to_override)
        self.state: StateConfig = set_sub_config('state', StateConfig, **configs_to_override)
        self.action_space: ActionSpaceConfig = set_sub_config('action_space', ActionSpaceConfig, **configs_to_override)
        self.reward: RewardConfig = set_sub_config('reward', RewardConfig, **configs_to_override)
        self.time_step: TimeStepConfig = set_sub_config('time_step', TimeStepConfig, **configs_to_override)
        self.zones: ZonesConfig = set_sub_config('zones', ZonesConfig, **configs_to_override)
        self.logger: LoggerConfig = set_sub_config('logger', LoggerConfig, 'environment', **configs_to_override)
        self.use_virtual_reset: bool = True
        super(EnvConfig, self).__init__(config_object_name='EnvConfig', root_dir=root_dir, **configs_to_override)
        self.export_exclude.append('logger')

    def _after_override_configs(self):
        self.action_space.quantity_movements = [i for i in range(self.trucks.capacity)]


def get_env_config(root_dir, config_path: str = None, log_level: Union[int, None] = None) -> EnvConfig:
    config = EnvConfig(root_dir)
    if config_path is not None:
        config_dict = load_config_dict(config_path, root_dir)
        config.set_configs(**config_dict)
    if log_level is not None:
        config.set_configs(logger={'level': log_level})
    return config
