import itertools
import os
from copy import deepcopy
from datetime import datetime
from typing import List, Union, Dict, Optional, Tuple, Any
from uuid import uuid4

from cabra.common.config import AbstractConfig, ConfigValueError, set_sub_config, load_config_dict, \
    ExportMode
from cabra.common.dict_utils import set_config_sub_value
from cabra.common.enum_utils import ExtendedEnum
from cabra.common.filesystem import ROOT_DIR
from cabra.core.step import Step
from cabra.common.data_structure import RunMode
from cabra.environment.agent import AgentType
from cabra.run.config import SingleRunConfig, MultiRunParamConfig

AGENT_WITH_VALIDATION = []

AGENT_ON_POLICY = [
    AgentType.PPO,
    AgentType.ConstrainedPPO
]

AGENT_ALIAS = {}


class HyperParamType(str, ExtendedEnum):
    Agent = 'agent'
    ZoneAgent = 'zone-agent'
    Environment = 'env'
    Run = 'run'
    Root = 'root'


class HyperParamValueMode(str, ExtendedEnum):
    Array = 'array'
    MultiArray = 'multi-array'


class RandomSeedsConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.run: List[int] = [10, 11, 12, 13, 14]
        super(RandomSeedsConfig, self).__init__('RandomSeedsConfig', **configs_to_override)


class AgentHyperParamConfig(AbstractConfig):

    def __init__(self, **configs_to_override):
        self.type: HyperParamType = HyperParamType.Root
        self.key: Optional[str] = 'param.key'
        self.values_mode: HyperParamValueMode = HyperParamValueMode.Array
        self.values: List[Union[str, int, float, Dict[str, Union[str, int, float]]]] = []
        super(AgentHyperParamConfig, self).__init__('AgentHyperParamConfig', **configs_to_override)

    def _after_override_configs(self):
        if isinstance(self.type, str):
            try:
                self.type: HyperParamType = HyperParamType(self.type)
            except Exception:
                raise ConfigValueError('type', self.type, module=self.name(),
                                       extra_msg=f'Possible values are: {HyperParamType.list()}')
        if isinstance(self.values_mode, str):
            try:
                self.values_mode: HyperParamValueMode = HyperParamValueMode(self.values_mode)
            except Exception:
                raise ConfigValueError('values_mode', self.values_mode, module=self.name(),
                                       extra_msg=f'Possible values are: {HyperParamValueMode.list()}')


class MultiRunConfig(AbstractConfig):

    def __init__(self, root_dir, **configs_to_override):
        self.base_run_config: SingleRunConfig = set_sub_config('base_run_config', SingleRunConfig,
                                                               root_dir,
                                                               **configs_to_override)
        self.random_seeds: RandomSeedsConfig = set_sub_config('random_seeds', RandomSeedsConfig, **configs_to_override)
        self.hyperparameters: Dict[AgentType, List[AgentHyperParamConfig]] = {
            # AgentType.DoubleDQN: [
            #     AgentHyperParamConfig(
            #         type=HyperParamType.Run,
            #         key='use_best_validation',
            #         values_mode=HyperParamValueMode.Array,
            #         values=[True, False]
            #     ),
            #     AgentHyperParamConfig(
            #         type=HyperParamType.Agent,
            #         key='double_dqn.learning_rate',
            #         values_mode=HyperParamValueMode.Array,
            #         values=[0.01, 0.005, 0.001]
            #     ),
            #     AgentHyperParamConfig(
            #         type=HyperParamType.Agent,
            #         key='double_dqn.q_net_addition_parameters.hidden_units',
            #         values_mode=HyperParamValueMode.Array,
            #         values=[
            #             [256, 128, 128, 64],
            #             [128, 64, 32, 16],
            #         ]
            #     )
            # ],
            AgentType.ConstrainedPPO: [
                AgentHyperParamConfig(
                    type=HyperParamType.Agent,
                    key='ppo.lr',
                    values_mode=HyperParamValueMode.Array,
                    values=[0.0001, 0.00005, 0.000001]
                ),
                AgentHyperParamConfig(
                    type=HyperParamType.Agent,
                    key='ppo.n_steps',
                    values_mode=HyperParamValueMode.Array,
                    values=[2048, 4096, 8192]
                ),
                AgentHyperParamConfig(
                    type=HyperParamType.Agent,
                    key='ppo.batch_size',
                    values_mode=HyperParamValueMode.Array,
                    values=[128, 256, 512]
                )
            ],
            AgentType.Random: [
                AgentHyperParamConfig(
                    type=HyperParamType.Root,
                    values_mode=HyperParamValueMode.MultiArray,
                    values=[
                        {'random_seeds.evaluation': [seed]}
                        for seed in [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
                    ]
                )
            ],
            AgentType.DoNothing: [
                AgentHyperParamConfig(
                    type=HyperParamType.Root,
                    values_mode=HyperParamValueMode.MultiArray,
                    values=[
                        {'random_seeds.evaluation': [seed]}
                        for seed in [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
                    ]
                )
            ],
            AgentType.ConstrainedRandom: [
                AgentHyperParamConfig(
                    type=HyperParamType.Root,
                    values_mode=HyperParamValueMode.MultiArray,
                    values=[
                        {
                            'random_seeds.evaluation': [seed],
                            'environment.constrained_space.max_distance': val,
                            'environment.constrained_space.zone_max_distance': val
                        } for val in [1000, 3000, 10000]
                        for seed in [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
                    ]
                )
            ],
            AgentType.ConstrainedGreedy: [
                AgentHyperParamConfig(
                    type=HyperParamType.Root,
                    values_mode=HyperParamValueMode.MultiArray,
                    values=[
                        {
                            'random_seeds.evaluation': [seed],
                            'environment.constrained_space.max_distance': val,
                            'environment.constrained_space.zone_max_distance': val
                        } for val in [1000, 3000, 10000]
                        for seed in [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
                    ]
                )
            ]
        }
        self.multi_run_name: str = '*auto*'
        self.skip_name_date: bool = False
        self.enforce_same_agent: bool = True
        super(MultiRunConfig, self).__init__('MultiRunConfig', **configs_to_override)

    def _after_override_configs(self):
        if self.multi_run_name == '*auto*' or self.multi_run_name is None:
            self.multi_run_name = str(uuid4())
        if isinstance(self.hyperparameters, dict):
            hyperparameters: Dict[AgentType, List[AgentHyperParamConfig]] = {}
            for agent_name, conf_list in self.hyperparameters.items():
                agent_type = agent_name if isinstance(agent_name, AgentType) else AgentType(agent_name)
                if agent_type not in hyperparameters:
                    hyperparameters[agent_type] = []
                if isinstance(conf_list, list):
                    for conf in conf_list:
                        if isinstance(conf, AgentHyperParamConfig):
                            hyperparameters[agent_type].append(conf)
                        elif isinstance(conf, dict):
                            hyperparameters[agent_type].append(AgentHyperParamConfig(**conf))
                        else:
                            raise ConfigValueError('hyperparameters', self.hyperparameters, self.name(),
                                                   extra_msg='Hyperaparameters entries must be an object')
                else:
                    raise ConfigValueError('hyperparameters', self.hyperparameters, self.name(),
                                           extra_msg='Hyperaparameters must be'
                                                     ' "Dict[AgentType, List[AgentHyperParamConfig]]"')
            self.hyperparameters: Dict[AgentType, List[AgentHyperParamConfig]] = hyperparameters

    def generate_runs_config(self) -> List[SingleRunConfig]:
        # TODO verify this
        runs: List[SingleRunConfig] = []
        scheduled_at = f'scheduled_at={datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
        for run_seed in self.random_seeds.run:
            for agent, agent_hp_config in self.hyperparameters.items():
                agent_combinations: List[
                    Tuple[Tuple[str, Any, HyperParamType], ...]] = generate_combinations(agent_hp_config)
                for param_combination in agent_combinations:
                    # create a new single_run_config
                    base_config = deepcopy(self.base_run_config.export(mode=ExportMode.DICT))
                    run_config = SingleRunConfig(root_dir=self.root_dir, **base_config)
                    run_config.random_seeds.training = run_seed
                    run_config.multi_run.is_multi_run = True
                    if self.skip_name_date is False:
                        run_config.multi_run.multi_run_code = f'{self.multi_run_name}-{scheduled_at}'
                    else:
                        run_config.multi_run.multi_run_code = self.multi_run_name
                    run_config.multi_run.multi_run_params.append(MultiRunParamConfig(
                        key=build_filename_key('seed'),
                        key_short=build_filename_key('seed'),
                        value=run_seed,
                        filename_key_val=f'seed={run_seed}'
                    ))
                    run_config.environment.agent.type = agent
                    if self.enforce_same_agent:
                        run_config.environment.zones.agent.type = agent
                    for param in param_combination:
                        if isinstance(param[0], tuple):
                            for sub_param in param:
                                update_single_run_config_param(run_config, sub_param)
                        else:
                            update_single_run_config_param(run_config, param)
                    runs.append(run_config)
        return runs

    @classmethod
    def generate_evaluation_config(cls,
                                   best_runs: List[dict],
                                   env_random_seed,
                                   agents: List[AgentType],
                                   batch_size: int,
                                   stop_step: Optional[Step] = None,
                                   step_size: Optional[int] = None,
                                   use_ssh_tunnel: bool = False,
                                   skip_name_date: bool = False,
                                   use_on_policy_agent: bool = True,
                                   seed_multiplier: int = 100,
                                   ):
        # TODO verify this
        if len(best_runs):
            base_config: SingleRunConfig = SingleRunConfig(root_dir=ROOT_DIR, **best_runs[0]['config'])
            base_config.run.run_mode = RunMode.Eval
            if stop_step is not None:
                base_config.run.stop_date = stop_step
            if step_size is not None:
                base_config.run.step_size = step_size
            base_config.run.bootstrapping_steps = 0
            base_config.run.rollout_batch_size = batch_size
            base_config.run.use_on_policy_agent = use_on_policy_agent
            base_config.multi_run.multi_run_params = []
            hyperparameters: Dict[str, List[dict]] = {}
            for agent_type in agents:
                agent_values: List[Dict[str, Union[int, float, str]]] = []
                for run in best_runs:
                    run_config: SingleRunConfig = SingleRunConfig(root_dir=ROOT_DIR, **run['config'])
                    if run_config.environment.agent.type == agent_type:
                        agent_type_str = agent_type.value.replace('-', '_')
                        if agent_type_str in AGENT_ALIAS:
                            agent_type_str = AGENT_ALIAS[agent_type_str]
                        base_config.environment.agent[agent_type_str] = run_config.environment.agent[agent_type_str]
                        base_config.saver.stats_condensed = False
                        base_config.saver.save_agent = False
                        run_seed = run_config.random_seeds.training
                        if agent_type in AGENT_WITH_VALIDATION:
                            model_path_base = run['save_folder_path']['path']
                            validation_index = run['validation_run_code'].replace(f'{run["run_code"]}_', '')
                            validation_name = f'validation_run_agent_state.pth'
                            model_path_full = os.path.join(model_path_base, f'validation_run-{validation_index}',
                                                           validation_name)
                            save_mode = run['save_folder_path']['save_mode']
                        else:
                            model_path_base = run['result_path']['path']
                            model_path_full = os.path.join(model_path_base, 'agent_state.pth')
                            save_mode = run['result_path']['save_mode']
                        if save_mode == 'minio':
                            base_path = run_config.saver.default_bucket
                        else:
                            base_path = run_config.saver.base_path
                        agent_values.append({
                            'random_seeds.training': run_seed * seed_multiplier,
                            'environment.agent.model_load.load_model': True,
                            'environment.agent.model_load.load_model_config.path': model_path_full,
                            'environment.agent.model_load.load_model_config.mode': save_mode,
                            'environment.agent.model_load.load_model_config.base_path': base_path,
                            'environment.agent.model_load.load_model_config.use_ssh_tunnel': use_ssh_tunnel
                        })
                hyperparameters[agent_type.value] = [
                    AgentHyperParamConfig(
                        type=HyperParamType.Root,
                        values_mode=HyperParamValueMode.MultiArray,
                        values=agent_values
                    ).export(mode=ExportMode.DICT)
                ]
            multi_run_config = {
                'base_run_config': base_config.export(mode=ExportMode.DICT),
                'random_seeds': {'run': [0]},
                'hyperparameters': hyperparameters,
                'multi_run_name': f'{base_config.multi_run.multi_run_code}-evaluation',
                'skip_name_date': skip_name_date,
            }
            return cls(root_dir=ROOT_DIR, **multi_run_config)
        else:
            raise AttributeError('Best runs array is empty')


def add_agent_model_params(
        container: Dict[str, Any],
        agent_type: AgentType,
        run: Dict[str, Any],
        run_config: SingleRunConfig,
        use_ssh_tunnel: bool,
        sub_agent_name: str
):
    if agent_type in AGENT_WITH_VALIDATION:
        model_path_base = run['save_folder_path']['path']
        validation_index = run['validation_run_code'].replace(f'{run["run_code"]}_', '')
        validation_name = f'validation_run_agent_state.pth'
        model_path_full = os.path.join(model_path_base, f'validation_run-{validation_index}',
                                       validation_name)
        save_mode = run['save_folder_path']['save_mode']
        if save_mode == 'minio':
            base_path = run_config.saver.default_bucket
        else:
            base_path = run_config.saver.base_path
    else:
        model_path_base = run['result_path']['path']
        model_path_full = os.path.join(model_path_base, 'agent_state.pth')
        save_mode = run['result_path']['save_mode']
        if save_mode == 'minio':
            base_path = run_config.saver.default_bucket
        else:
            base_path = run_config.saver.base_path

    container[f'environment.agent.model_load.{sub_agent_name}.path'] = model_path_full
    container[f'environment.agent.model_load.{sub_agent_name}.mode'] = save_mode
    container[f'environment.agent.model_load.{sub_agent_name}.base_path'] = base_path
    container[f'environment.agent.model_load.{sub_agent_name}.use_ssh_tunnel'] = use_ssh_tunnel


def get_multi_run_config(root_dir, config_path: str = None) -> MultiRunConfig:
    config = MultiRunConfig(root_dir)
    if config_path is not None:
        config_dict = load_config_dict(config_path, root_dir)
        config.set_configs(**config_dict)
    return config


def generate_combinations(
        parameters: List[AgentHyperParamConfig]) -> List[Tuple[Tuple[str, Any, HyperParamType], ...]]:
    possibilities: List[List[Tuple[str, Any, HyperParamType]]] = []
    multi_param_possibilities: List[Tuple[Tuple[str, Any, HyperParamType], ...]] = []
    multi_param_sub_list = []
    # I need to use itertools.combinatinos to generate all the possible combinations
    for param_config in parameters:
        if param_config.values_mode == HyperParamValueMode.Array:
            param_possibilities: List[Tuple[str, Any, HyperParamType]] = []
            for value in param_config.values:
                param_possibilities.append(
                    (param_config.key, value, param_config.type)
                )
            possibilities.append(param_possibilities)
        elif param_config.values_mode == HyperParamValueMode.MultiArray:
            sub_list = []
            for dict_values in param_config.values:
                sub_param: List[Tuple[str, Any, HyperParamType]] = []
                for key, value in dict_values.items():
                    sub_param.append(
                        (key, value, param_config.type)
                    )
                multi_param_possibilities.append(tuple(sub_param))
                sub_list.append(tuple(sub_param))
            multi_param_sub_list.append(sub_list)
        else:
            raise ValueError('Parameter values_mode is invalid')
    if len(multi_param_possibilities) > 0 and len(possibilities) == 0:
        return multi_param_possibilities
    elif len(multi_param_possibilities) == 0 and len(possibilities) > 0:
        return list(itertools.product(*possibilities))
    else:
        multi_combined_pre = []
        for sub_multi in multi_param_sub_list:
            multi_combined_pre += list(itertools.combinations(sub_multi, len(sub_multi)))
        multi_combined = list(itertools.product(*multi_combined_pre))
        poss_product = list(itertools.product(*possibilities))
        final = []
        for multi in multi_combined:
            for poss in poss_product:
                final.append(multi + poss)
        return final


SHORT_NAME_MAPPING = {
    'learning_rate': 'lr',
    'target_net_update_frequency': 'target-update',
}


def build_filename_key(key):
    key_parts = key.split('.')
    prefix = ''
    if 'agent' in key:
        prefix = 'agent'
    if 'zones.agent' in key:
        prefix = 'zone_agent'
    filename_key = key_parts[-1]
    if key == 'random_seeds.run':
        return 'run-seed'
    if key == 'environment.agent.model_load.path':
        return 'model-path'
    if filename_key in SHORT_NAME_MAPPING:
        filename_key = SHORT_NAME_MAPPING[filename_key]
    else:
        filename_key = filename_key.replace('_', '-')
    if len(prefix) > 0:
        return f'{prefix}-{filename_key}'
    else:
        return filename_key


def build_filename_key_val(key: str, value: Any) -> str:
    filename_key = build_filename_key(key)
    if isinstance(value, list):
        filename_value = ','.join([str(v) for v in value])
    else:
        filename_value = str(value)
    return f'{filename_key}={filename_value}'


HYPERPARAMETER_TYPE_MAPPING = {
    HyperParamType.Root: '',
    HyperParamType.Environment: 'environment',
    HyperParamType.Agent: 'environment.agent',
    HyperParamType.ZoneAgent: 'environment.zones.agent',
    HyperParamType.Run: 'run',
}


def update_single_run_config_param(config: SingleRunConfig, param: Tuple[str, Any, HyperParamType]):
    key, value, param_type = param
    key_start = HYPERPARAMETER_TYPE_MAPPING[param_type]
    if len(key_start) > 0:
        full_key = f'{key_start}.{key}'
    else:
        full_key = key
    set_config_sub_value(config, key=full_key, value=value)
    multi_run_param: MultiRunParamConfig = MultiRunParamConfig(
        key=build_filename_key(full_key),
        key_short=build_filename_key(full_key),
        value=value,
        filename_key_val=build_filename_key_val(full_key, value)
    )
    if multi_run_param.key == 'agent-seed':
        first_param = config.multi_run.multi_run_params[0]
        if first_param.key == build_filename_key('seed'):
            del config.multi_run.multi_run_params[0]
    config.multi_run.multi_run_params.append(multi_run_param)
