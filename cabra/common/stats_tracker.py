from dataclasses import dataclass
from collections import deque
from typing import Optional, Dict, List, Tuple, Union, Any

import numpy as np

from cabra import SingleRunConfig, ROOT_DIR
from cabra.common.config import ExportMode
from cabra.common.data_structure import RunMode
from cabra.common.tensorboard_wrapper import TensorboardWrapper
from cabra.environment.node import Node


def last_item(arr):
    return arr[-1]


@dataclass
class VariableInfo:
    tensorboard: bool = False
    is_only_cumulative: bool = False
    is_string: bool = False
    keep_cumulative: bool = False
    disk_save: bool = True
    db_save: bool = False
    redis_save: bool = False
    aggregation_fn: Optional[callable] = None
    single_value_is_array: bool = False
    keep_every_x: Optional[int] = None
    sliding_window: Optional[int] = None
    str_precision: int = 4
    is_constant: bool = False
    is_running_value: bool = False


class Tracker:

    def __init__(
            self,
            run_code: str = None,
            config: Optional[SingleRunConfig] = None,
            run_mode: Optional[RunMode] = None,
            tensorboard: Optional[TensorboardWrapper] = None
    ):
        self.config: SingleRunConfig = config
        self.run_code: str = run_code
        self.run_mode: RunMode = run_mode
        self.tensorboard: Optional[TensorboardWrapper] = tensorboard
        self.is_condensed: bool = True
        # tracking props
        self.history_suffix = '/history'
        self.cumulative_suffix = '/cumulative'
        self._tracked: Dict[str, Any] = {}
        self._keep_tmp: Dict[str, Union[List[Union[int, float, str, bool]], Union[int, float, str]]] = {}
        self._keep_tmp_n_updates: Dict[str, int] = {}
        self._variables_info: Dict[str, VariableInfo] = {}
        self._last_tracked_step: Dict[str, Optional[int]] = {}
        self._running_tracked: Dict[str, Any] = {}
        # public props
        self.rollout_variables: List[str] = ['total_steps', 'reward', 'shortages', 'env_shortages', 'cost',
                                             'solved_steps', 'penalties', 'empty_steps']
        self.rollout_variables_eval: List[str] = self.rollout_variables
        self.rollout_variables_eval_step: List[str] = ['step/reward', 'step/shortages',
                                                       'step/env_shortages', 'step/cost', 'step/solved_steps']
        self.episode_variables: List[str] = ['total_steps', 'reward', 'shortages', 'env_shortages', 'cost',
                                             'solved_steps', 'penalties', 'empty_steps']

    def __str__(self):
        return f'<StatsTracker tracked_variables={len(self._variables_info)}>'

    def __len__(self):
        return len(self._variables_info)

    def _track_value(self, key: str, value: Union[int, float, str, bool, np.ndarray], step: Optional[int] = None):
        variable_info = self._variables_info[key]
        # prevent multiple tracking of the same variable at the same step
        if step is not None and step == self._last_tracked_step[key]:
            return
        if variable_info.is_running_value:
            self._track_running_value(key, value, variable_info)
            return
        if variable_info.is_constant:
            self._tracked[key] = value
        else:
            self._last_tracked_step[key] = step
            add_value = True
            if variable_info.keep_every_x is not None:
                if variable_info.single_value_is_array:
                    self._keep_tmp[key] += value.tolist()
                else:
                    self._keep_tmp[key] += value
                self._keep_tmp_n_updates[key] += 1
                if self._keep_tmp_n_updates[key] == variable_info.keep_every_x:
                    add_value = True
                    if variable_info.single_value_is_array:
                        value = np.array(self._keep_tmp[key])
                        self._keep_tmp[key] = []
                    else:
                        value = self._keep_tmp[key]
                        self._keep_tmp[key] = 0
                    self._keep_tmp_n_updates[key] = 0
                else:
                    add_value = False
            if add_value:
                if variable_info.single_value_is_array:
                    self._track_value(f'{key}/total', value.sum(), step)
                    self._track_value(f'{key}/avg', value.mean(), step)
                    self._track_value(f'{key}/std', value.std(), step)
                    self._track_value(f'{key}/min', value.min(), step)
                    self._track_value(f'{key}/max', value.max(), step)
                else:
                    if variable_info.is_only_cumulative:
                        self._tracked[key] += value
                    elif variable_info.sliding_window:
                        self._tracked[key].append(value)
                    elif variable_info.is_string:
                        self._tracked[key] = value
                    elif variable_info.keep_cumulative:
                        self._tracked[key + self.history_suffix].append(value)
                        self._tracked[key + self.cumulative_suffix] += value
                    else:
                        self._tracked[key].append(value)
                    if variable_info.tensorboard and step is not None:
                        self._tensorboard_track(key, value, step)

    def _track_running_value(self, key: str, value: Any, variable_info: VariableInfo):
        if variable_info.is_only_cumulative:
            self._running_tracked[key] += value
        elif variable_info.keep_cumulative:
            self._running_tracked[key + self.history_suffix].append(value)
            self._running_tracked[key + self.cumulative_suffix] += value
        else:
            self._running_tracked[key].append(value)

    def _tensorboard_track(self, key: str, value: Union[int, float, str, bool], step: int):
        if self.tensorboard is not None:
            variable_info = self._variables_info[key]
            if variable_info.is_constant:
                return
            if variable_info.keep_cumulative:
                self.tensorboard.add_scalar(tag=key + self.history_suffix, value=value, step=step)
                self.tensorboard.add_scalar(tag=key + self.cumulative_suffix,
                                            value=self._tracked[key + self.cumulative_suffix], step=step)
            elif variable_info.is_only_cumulative:
                self.tensorboard.add_scalar(tag=key, value=self._tracked[key], step=step)
            else:
                self.tensorboard.add_scalar(tag=key, value=value, step=step)

    def init_tracking(self, key: str, initial_value: Optional[Any] = None, **variable_info):
        if key not in self._variables_info:
            if key.endswith(self.cumulative_suffix):
                raise KeyError(f'trying to track a new variable using protected suffix "{self.cumulative_suffix}"')
            if key.endswith(self.history_suffix):
                raise KeyError(f'trying to track a new variable using protected suffix "{self.history_suffix}"')
            variable_info = VariableInfo(**variable_info)
            self._variables_info[key] = variable_info
            self._last_tracked_step[key] = None
            if variable_info.is_running_value:
                self._init_running_value_variable(key, variable_info, initial_value)
            else:
                if variable_info.single_value_is_array:
                    self._init_single_value_is_array_variable(key, variable_info)
                if variable_info.keep_every_x is not None:
                    self._init_keep_every_x_variable(key, variable_info)
                elif variable_info.sliding_window is not None:
                    self._init_sliding_window_variable(key, variable_info)
                else:
                    if variable_info.is_only_cumulative:
                        self._init_is_only_cumulative_variable(key, variable_info, initial_value)
                    elif variable_info.keep_cumulative:
                        self._init_keep_cumulative_variable(key, variable_info, initial_value)
                    else:
                        self._init_variable(key, variable_info, initial_value)

    def _init_single_value_is_array_variable(self, key: str, variable_info: VariableInfo):
        self.init_tracking(f'{key}/total', tensorboard=variable_info.tensorboard,
                           db_save=variable_info.db_save, redis_save=variable_info.redis_save,
                           aggregation_fn=np.max)
        self.init_tracking(f'{key}/avg', tensorboard=variable_info.tensorboard,
                           db_save=variable_info.db_save, redis_save=variable_info.redis_save,
                           aggregation_fn=np.max)
        self.init_tracking(f'{key}/std', tensorboard=variable_info.tensorboard,
                           db_save=variable_info.db_save, redis_save=variable_info.redis_save,
                           aggregation_fn=np.max)
        self.init_tracking(f'{key}/min', tensorboard=variable_info.tensorboard)
        self.init_tracking(f'{key}/max', tensorboard=variable_info.tensorboard)

    def _init_keep_every_x_variable(self, key: str, variable_info: VariableInfo):
        if variable_info.single_value_is_array:
            self._keep_tmp[key] = []
        else:
            self._keep_tmp[key] = 0
            self._tracked[key] = []
        self._keep_tmp_n_updates[key] = 0

    def _init_sliding_window_variable(self, key: str, variable_info: VariableInfo):
        self._tracked[key] = deque([], maxlen=variable_info.sliding_window)

    def _init_is_only_cumulative_variable(self, key: str, variable_info: VariableInfo, initial_value=None):
        self._tracked[key] = 0
        if initial_value:
            self._tracked[key] = initial_value

    def _init_keep_cumulative_variable(self, key: str, variable_info: VariableInfo, initial_value=None):
        self._tracked[key + self.history_suffix] = []
        self._tracked[key + self.cumulative_suffix] = 0
        if initial_value:
            self._tracked[key + self.history_suffix] = initial_value[0]
            self._tracked[key + self.cumulative_suffix] = initial_value[1]

    def _init_variable(self, key: str, variable_info: VariableInfo, initial_value=None):
        self._tracked[key] = []
        if initial_value:
            self._tracked[key] = initial_value

    def _init_running_value_variable(self, key: str, variable_info: VariableInfo, initial_value=None):
        variable_info.disk_save = False
        variable_info.db_save = False
        variable_info.redis_save = False
        self._variables_info[key] = variable_info
        if variable_info.is_only_cumulative:
            self._running_tracked[key] = 0 if initial_value is None else initial_value
        elif variable_info.keep_cumulative:
            self._running_tracked[key + self.history_suffix] = [] if initial_value is None else initial_value[0]
            self._running_tracked[key + self.cumulative_suffix] = 0 if initial_value is None else initial_value[1]
        else:
            self._running_tracked[key] = [] if initial_value is None else initial_value

    def reset_running_variable(self, key: str, reset_value: Any = None):
        variable_info = self._variables_info[key]
        self._init_running_value_variable(key, variable_info, reset_value)

    def reset_multiple_variables(self, keys: List[str], prefix: str):
        for var in keys:
            self.reset_running_variable(f'{prefix}{var}')

    def track(
            self,
            key: str,
            value: Union[int, float, str, bool, None, np.ndarray],
            step: Optional[int] = None,
            replace_value: bool = False,
            **variable_info):
        if key not in self._variables_info:
            # init the variable
            self.init_tracking(key, **variable_info)
        # track the value
        if not replace_value:
            self._track_value(key, value, step)
        else:
            self._tracked[key] = value

    def track_running_variables(self, prefix: str, **variables):
        for key, value in variables.items():
            self.track(f'{prefix}{key}', value)

    def track_running_rollout_variables(
            self,
            prefix: str,
            reward,
            shortages,
            env_shortages,
            cost,
            solved_steps,
            penalties,
            empty_steps,
            total_steps,
            choose_time=None
    ):
        if choose_time is None:
            self.track_running_variables(prefix, reward=reward, shortages=shortages, env_shortages=env_shortages,
                                         cost=cost, solved_steps=solved_steps, penalties=penalties,
                                         empty_steps=empty_steps, total_steps=total_steps)
        else:
            self.track_running_variables(prefix, reward=reward, shortages=shortages, env_shortages=env_shortages,
                                         cost=cost, solved_steps=solved_steps, penalties=penalties,
                                         empty_steps=empty_steps, total_steps=total_steps, choose_time=choose_time)

    def track_running_episode_variables(
            self,
            prefix: str,
            reward,
            shortages,
            env_shortages,
            cost,
            solved_steps,
            penalties,
            empty_steps,
            total_steps
    ):
        self.track_running_variables(prefix, reward=reward, shortages=shortages, env_shortages=env_shortages,
                                     cost=cost, solved_steps=solved_steps, penalties=penalties,
                                     empty_steps=empty_steps, total_steps=total_steps)

    def track_running_episode_and_rollout_variables(
            self,
            rollout_prefix: str,
            episode_prefix: str,
            is_eval: bool,
            reward,
            shortages,
            env_shortages,
            cost,
            solved_steps,
            penalties,
            empty_steps,
            total_steps,
    ):
        self.track_running_rollout_variables(
            rollout_prefix, reward, shortages, env_shortages, cost, solved_steps, penalties, empty_steps, total_steps)
        if not is_eval:
            self.track_running_episode_variables(
                episode_prefix, reward, shortages, env_shortages, cost,
                solved_steps, penalties, empty_steps, total_steps)

    def track_eval_step_metrics(self, prefix: str, step_metrics: Dict[str, float]):
        for key in self.rollout_variables_eval_step:
            self.track(f'{prefix}{key}', step_metrics[key])

    def track_episode_end(self, running_prefix: str, prefix: str):
        for key in self.episode_variables:
            value = self.get(f'{running_prefix}{key}')
            self.track(f'{prefix}{key}', value)
        self.reset_multiple_variables(self.episode_variables, running_prefix)

    def track_rollout_end(
            self,
            running_prefix: str,
            train_prefix: Optional[str] = None,
            eval_array_prefix: Optional[str] = None,
            eval_history_prefix: Optional[str] = None,
            is_eval: bool = False,
    ):
        variables = self.rollout_variables if not is_eval else self.rollout_variables_eval
        for key in variables:
            if not is_eval:
                value = self.get(f'{running_prefix}{key}')
                self.track(f'{train_prefix}{key}', value)
            else:
                history = self.get(f'{running_prefix}{key}')
                self.track(f'{eval_array_prefix}{key}', np.array(history))
                self.track(f'{eval_history_prefix}{key}', history, replace_value=True)
        self.reset_multiple_variables(variables, running_prefix)
        if is_eval:
            for key in self.rollout_variables_eval_step:
                history = self.get(f'{running_prefix}{key}')
                self.track(f'{eval_history_prefix}{key}', history, replace_value=True)
            self.reset_multiple_variables(self.rollout_variables_eval_step, running_prefix)

    def disk_stats(self) -> Dict[str, Any]:
        stats = {
            'run_code': self.run_code,
            'run_mode': self.run_mode,
            'is_condensed': self.is_condensed,
            'config': self.config.export(mode=ExportMode.DICT)
        }
        for key, variable_info in self._variables_info.items():
            if variable_info.disk_save:
                if not variable_info.single_value_is_array and not variable_info.is_running_value:
                    if variable_info.keep_cumulative:
                        stats[key + self.history_suffix] = self._tracked[key + self.history_suffix]
                        stats[key + self.cumulative_suffix] = self._tracked[key + self.cumulative_suffix]
                    elif variable_info.sliding_window:
                        stats[key] = list(self._tracked[key])
                    else:
                        stats[key] = self._tracked[key]
        return stats

    def get_key_to_string(self, key: str, aggr_fn: Optional[callable] = None, **kwargs) -> str:
        value = self.get(key, **kwargs)
        if aggr_fn is not None:
            value = aggr_fn(value)
        if value is None:
            raise AttributeError(f'Key {key} not exists in tracked variables')
        variable_info = self._variables_info[key]
        return str(round(value, variable_info.str_precision))

    def _get_aggregable_stats(self, key: str, variable_info: VariableInfo, aggr_fn: Optional[callable] = None) -> Any:
        if variable_info.keep_cumulative:
            return self._tracked[key + self.cumulative_suffix]
        elif variable_info.is_only_cumulative:
            return self._tracked[key]
        elif variable_info.is_constant:
            return self._tracked[key]
        else:
            aggregation_fn = variable_info.aggregation_fn if aggr_fn is None else aggr_fn
            if aggregation_fn is not None:
                if len(self._tracked[key]) > 0:
                    return aggregation_fn(self._tracked[key])
                else:
                    return 0

    def db_stats(self, add_config=False, add_run_code=False, add_run_mode=False,
                 best_prefix: Optional[str] = None, best_key: Optional[str] = None,
                 rename_key: Optional[str] = None) -> Dict[str, Any]:
        stats = {}
        if add_config:
            stats['config'] = self.config
        if add_run_code:
            stats['run_code'] = self.run_code
        if add_run_mode:
            stats['run_mode'] = self.run_mode
        best_index = None
        if best_key is not None:
            value = self.get(best_key)
            if len(value) > 0:
                best_index = np.argmax(value)
        for key, variable_info in self._variables_info.items():
            if variable_info.db_save and not variable_info.single_value_is_array:
                stats_key = key
                if best_prefix is not None and best_prefix in key and best_index is not None:
                    value = self._tracked[key][best_index]
                    if rename_key is not None:
                        stats_key = key.replace(best_prefix, rename_key)
                else:
                    value = self._get_aggregable_stats(key, variable_info)
                if hasattr(value, 'item'):
                    value = value.item()
                stats[stats_key] = value
        return stats

    def redis_stats(self,
                    best_prefix: Optional[str] = None,
                    best_key: Optional[str] = None,
                    rename_key: Optional[str] = None,
                    get_max_and_last: bool = False,
                    ) -> Dict[str, Any]:
        stats = {}
        best_index = None
        if best_key is not None:
            value = self.get(best_key)
            if len(value) > 0:
                best_index = np.argmax(value)
        for key, variable_info in self._variables_info.items():
            if variable_info.redis_save and not variable_info.single_value_is_array:
                if best_prefix is not None and best_prefix in key and best_index is not None:
                    if rename_key is not None:
                        stats_key = key.replace(best_prefix, rename_key)
                    else:
                        stats_key = key
                    stats[stats_key] = self._tracked[key][best_index]
                else:
                    if get_max_and_last and str(RunMode.Eval.value) in key:
                        stats[f'{key}-max'] = self._get_aggregable_stats(key, variable_info, aggr_fn=np.max)
                        stats[f'{key}-last'] = self._get_aggregable_stats(key, variable_info, aggr_fn=last_item)
                    else:
                        stats[key] = self._get_aggregable_stats(key, variable_info)
        return stats

    def get(self,
            key: str,
            include_info=False,
            add_cumulative=False,
            only_cumulative=False) -> Optional[Union[Any, Tuple[Any, VariableInfo]]]:
        result = None
        if key in self._variables_info:
            variable_info = self._variables_info[key]
            if variable_info.is_constant and key not in self._tracked:
                result = None
            if variable_info.is_running_value:
                if variable_info.keep_cumulative:
                    result = self._running_tracked[key + self.history_suffix]
                    if add_cumulative:
                        result = {
                            key + self.history_suffix: result,
                            key + self.cumulative_suffix: self._running_tracked[key + self.cumulative_suffix]
                        }
                    elif only_cumulative:
                        result = self._running_tracked[key + self.cumulative_suffix]
                else:
                    result = self._running_tracked[key]
                return result
            else:
                if variable_info.keep_cumulative:
                    result = self._tracked[key + self.history_suffix]
                    if add_cumulative:
                        result = {
                            key + self.history_suffix: result,
                            key + self.cumulative_suffix: self._tracked[key + self.cumulative_suffix]
                        }
                    elif only_cumulative:
                        result = self._tracked[key + self.cumulative_suffix]
                elif variable_info.sliding_window:
                    values = self._tracked[key]
                    if len(values) == variable_info.sliding_window:
                        result = sum(values) / variable_info.sliding_window
                    else:
                        result = None
                else:
                    result = self._tracked[key]
            if include_info:
                result = (result, variable_info)
        return result

    def get_array_data(self, key, index):
        if key in self._variables_info:
            variable_info = self._variables_info[key]
            if variable_info.single_value_is_array:
                return {
                    f'{key}/total': self.get(f'{key}/total')[index],
                    f'{key}/avg': self.get(f'{key}/avg')[index],
                    f'{key}/std': self.get(f'{key}/std')[index],
                    f'{key}/min': self.get(f'{key}/min')[index],
                    f'{key}/max': self.get(f'{key}/max')[index],
                }

    def __getitem__(self, item):
        return self._tracked[item]

    def __contains__(self, item):
        return item in self._tracked

    @classmethod
    def from_dict(cls, variables_dict: Dict[str, Any]) -> 'Tracker':
        tracker = cls(
            run_code=variables_dict['run_code'],
            run_mode=variables_dict['run_mode'],
            config=SingleRunConfig(root_dir=ROOT_DIR, **variables_dict['config'])
        )
        tracker.is_condensed = variables_dict['is_condensed']
        reserved_keys = ['run_code', 'run_mode', 'config']
        for key, value in variables_dict.items():
            if key not in reserved_keys:
                variable_info = {}
                var_key = key
                initial_value = value
                if key.endswith(tracker.history_suffix):
                    variable_info['keep_cumulative'] = True
                    var_key = key.replace(tracker.history_suffix, '')
                    initial_value = (value, variables_dict[var_key + tracker.cumulative_suffix])
                elif key.endswith(tracker.cumulative_suffix):
                    variable_info['keep_cumulative'] = True
                    var_key = key.replace(tracker.cumulative_suffix, '')
                    initial_value = (variables_dict[var_key + tracker.history_suffix], value)
                elif isinstance(value, (int, float)):
                    variable_info['is_only_cumulative'] = True
                tracker.init_tracking(var_key, initial_value=initial_value, **variable_info)
        return tracker

    @classmethod
    def init_condensed_tracker(cls, run_code: str, config: SingleRunConfig,
                               run_mode: Optional[RunMode] = None,
                               tensorboard: Optional[TensorboardWrapper] = None) -> 'Tracker':
        tracker = cls(run_code, config, run_mode, tensorboard)
        if config.environment.agent.type.is_on_policy():
            init_on_policy_tracker(tracker)
        elif config.environment.agent.type.is_off_policy():
            init_off_policy_tracker(tracker)
        else:
            init_baseline_tracker(tracker)
        return tracker

    @classmethod
    def init_tracker(cls, nodes: List[Node], run_code: str, config: SingleRunConfig,
                     run_mode: Optional[RunMode] = None,
                     tensorboard: Optional[TensorboardWrapper] = None) -> 'Tracker':
        tracker = Tracker.init_condensed_tracker(run_code, config, run_mode, tensorboard)
        for i, node in enumerate(nodes):
            tracker.init_tracking(f'nodes/n_{i}/bikes', tensorboard=True)
            tracker.init_tracking(f'nodes/n_{i}/empty_slots', tensorboard=True)
        tracker.is_condensed = False
        return tracker


def init_baseline_tracker(tracker: Tracker):
    init_evaluation_tracker(tracker)


def init_on_policy_tracker(tracker: Tracker):
    episode_window = tracker.config.run.metrics_window
    metrics_window = 1
    init_base_training_tracker(tracker, episode_window=episode_window, metrics_window=metrics_window)
    init_evaluation_tracker(tracker)


def init_off_policy_tracker(tracker: Tracker):
    episode_window = tracker.config.run.metrics_window
    metrics_window = tracker.config.run.metrics_window
    init_base_training_tracker(tracker, episode_window=episode_window, metrics_window=metrics_window)
    init_evaluation_tracker(tracker)


def init_evaluation_tracker(tracker: 'Tracker'):
    eval_seeds = tracker.config.random_seeds.evaluation
    eval_prefix = RunMode.Eval.value
    for seed in eval_seeds:
        for key in tracker.rollout_variables_eval:
            tracker.init_tracking(f'{eval_prefix}-{seed}/history/{key}', tensorboard=True)
            tracker.init_tracking(f'{eval_prefix}-{seed}/rollout_running/{key}',
                                  is_running_value=True, keep_cumulative=True)
            tracker.init_tracking(f'{eval_prefix}-{seed}/{key}', single_value_is_array=True)
            tracker.init_tracking(f'{eval_prefix}/avg/{key}', db_save=True, tensorboard=True, redis_save=True,
                                  aggregation_fn=last_item)
            tracker.init_tracking(f'{eval_prefix}/best_run/{key}', is_constant=True, db_save=True, redis_save=True)
        for key in tracker.rollout_variables_eval_step:
            tracker.init_tracking(f'{eval_prefix}-{seed}/history/{key}', tensorboard=True)
            tracker.init_tracking(f'{eval_prefix}-{seed}/rollout_running/{key}',
                                  is_running_value=True, keep_cumulative=True)


def init_base_training_tracker(tracker: Tracker, episode_window: int, metrics_window: int):
    prefix = RunMode.Train.value
    if tracker.run_mode == RunMode.Train:
        # training metrics
        tracker.init_tracking(f'{prefix}/episodes', is_only_cumulative=True, redis_save=True, tensorboard=True,
                              db_save=True, str_precision=0)
        tracker.init_tracking(f'{prefix}/training_steps', is_only_cumulative=True, redis_save=True, tensorboard=True,
                              db_save=True, str_precision=0)

        redis_exclude = ['total_steps', 'penalties', 'empty_steps']

        for key in tracker.rollout_variables:
            redis_save = True if key not in redis_exclude else False
            tracker.init_tracking(f'{prefix}/rollout_running/{key}', is_running_value=True, is_only_cumulative=True)
            tracker.init_tracking(f'{prefix}/avg-{metrics_window}-steps/{key}', keep_every_x=metrics_window,
                                  redis_save=redis_save, tensorboard=True, aggregation_fn=last_item)
        for key in tracker.episode_variables:
            redis_save = True if key not in redis_exclude else False
            tracker.init_tracking(f'{prefix}/episode_running/{key}', is_running_value=True, is_only_cumulative=True)
            tracker.init_tracking(f'{prefix}/episode/{key}', sliding_window=episode_window, redis_save=redis_save,
                                  tensorboard=True, aggregation_fn=np.mean)
