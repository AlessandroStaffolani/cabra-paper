from copy import deepcopy
from logging import Logger
from typing import Optional, Union, Dict, List

from numpy.random import RandomState
from torch.multiprocessing import Pool, set_start_method

from cabra import SingleRunConfig, single_run_config, logger, ROOT_DIR
from cabra.common.config import ExportMode
from cabra.common.data_structure import RunMode
from cabra.common.object_handler import create_object_handler, MinioObjectHandler, ObjectHandler
from cabra.common.stats_tracker import Tracker
from cabra.environment.agent.abstract import AgentAbstract
from cabra.environment.agent.factory import create_agent
from cabra.environment.truck import TrucksWrapper
from cabra.environment.zones_env_wrapper import ZonesEnvWrapper
from cabra.run.runner.rollout_sampler import track_average_values, \
    collect_rollout, collect_single_step, track_best_run_variables, track_weekly_evaluation
from cabra.run.runner.run_logger import RunLogger
from cabra.run.runner.utils import load_agent_model


def _return_new_env(
        config: SingleRunConfig,
        random_state: RandomState,
        run_code: str,
        run_mode: RunMode,
        random_seed: int,
        log: Logger
) -> ZonesEnvWrapper:
    env = ZonesEnvWrapper(
        run_code=run_code,
        config=config,
        random_state=random_state,
        log=log,
        run_mode=run_mode,
        random_seed=random_seed,
    )
    return env


def _init_eval_env(run_code: str, config_dict: dict, random_seed: int, mode: RunMode = RunMode.Eval, log: Logger = None):
    random_state = RandomState(random_seed)
    eval_config: SingleRunConfig = SingleRunConfig(
        ROOT_DIR,
        **config_dict
    )
    return _return_new_env(config=eval_config, random_state=random_state,
                           run_code=f'{run_code}_{mode.value}',
                           run_mode=mode,
                           random_seed=random_seed,
                           log=log)


def _return_new_agent(
        config: SingleRunConfig,
        env: ZonesEnvWrapper,
        random_seed: int,
        init_seed=True,
        run_mode=RunMode.Train,
        log: Logger = None
):
    agent = create_agent(
        agent_type=config.environment.agent.type,
        action_space=env.action_space,
        random_state=env.random,
        action_spaces=env.action_spaces,
        state_spaces=env.state_spaces,
        nodes=env.nodes,
        zones=env.zones,
        nodes_max_distance=env.nodes_max_distance,
        config=config,
        log=log,
        state_builder=env.state_builder,
        is_zone_agent=False,
        mode=run_mode,
        init_seed=init_seed,
        random_seed=random_seed,
        distances_provider=env.distances_provider,
        **config.environment.agent.global_parameters
    )
    if agent.name.is_oracle():
        agent.load_generator = env.load_generator
    return agent


def _return_new_zone_agent(
        config: SingleRunConfig,
        env: ZonesEnvWrapper,
        random_seed: int,
        init_seed=True,
        run_mode=RunMode.Train,
        log: Logger = None
):
    agent = create_agent(
        agent_type=config.environment.zones.agent.type,
        action_space=env.zone_action_space,
        random_state=env.random,
        action_spaces=env.action_spaces,
        state_spaces=env.state_spaces,
        nodes=env.nodes,
        zones=env.zones,
        nodes_max_distance=env.nodes_max_distance,
        config=config,
        log=log,
        state_builder=env.state_builder,
        is_zone_agent=True,
        mode=run_mode,
        init_seed=init_seed,
        random_seed=random_seed,
        distances_provider=env.distances_provider,
        **config.environment.agent.global_parameters
    )
    if agent.name.is_oracle():
        agent.load_generator = env.load_generator
    return agent


def _do_evaluations_async(
        seeds: List[int],
        agent_state_model,
        zone_agent_state_model,
        eval_counter: int,
        config_dict: dict,
        run_code: str,
):
    config = SingleRunConfig(ROOT_DIR, **config_dict)
    stats_tracker: Tracker = Tracker.init_condensed_tracker(
        run_code=run_code, config=config, run_mode=RunMode.Eval, tensorboard=None
    )
    eval_agent = None
    eval_zone_agent = None
    for i, seed in enumerate(seeds):
        eval_env = _init_eval_env(run_code, config_dict, random_seed=seed, mode=RunMode.Eval)
        eval_env.set_stats_tracker(stats_tracker)
        eval_trucks_wrapper = TrucksWrapper(eval_env.trucks, eval_env.config)
        eval_env.reset(show_log=False)
        eval_zone_agent = _return_new_zone_agent(
            config, eval_env, random_seed=seed, init_seed=False, run_mode=RunMode.Eval)
        eval_agent = _return_new_agent(config, eval_env, random_seed=seed, init_seed=False, run_mode=RunMode.Eval)
        eval_agent.set_stats_tracker(stats_tracker)
        eval_zone_agent.set_stats_tracker(stats_tracker)

        eval_agent.load_agent_state(agent_state_model)
        eval_zone_agent.load_agent_state(zone_agent_state_model)

        eval_agent.set_mode(RunMode.Eval)
        eval_agent.set_nodes(eval_env.nodes)

        n_episodes = config.run.evaluation_episodes
        if eval_env.load_generator.eval_every_week:
            n_episodes = 4

        collect_rollout(
            env=eval_env,
            agent=eval_agent,
            zone_agent=eval_zone_agent,
            trucks_wrapper=eval_trucks_wrapper,
            rollout_size=-1,
            tracker=stats_tracker,
            run_mode=RunMode.Eval,
            seed=seed,
            is_bootstrapping=False,
            n_episodes=n_episodes
        )

    track_average_values(tracker=stats_tracker, mode=RunMode.Eval, iteration=0, seeds=seeds)
    if config.emulator.model.cdrc_data_model.eval_every_week:
        track_weekly_evaluation(tracker=stats_tracker, seeds=seeds, n_weeks=4)
    eval_performance = stats_tracker.get(config.run.keep_metric)[-1]
    eval_metrics = {'iteration': eval_counter}
    for key in stats_tracker.rollout_variables_eval:
        value = stats_tracker.get(f'evaluation/avg/{key.replace("step/", "")}')[-1]
        eval_metrics[key.replace('step/', '')] = value
    eval_metrics['eval_agent'] = eval_agent.get_agent_state()
    eval_metrics['eval_performance'] = eval_performance
    eval_metrics['eval_counter'] = eval_counter
    eval_metrics['eval_zone_agent'] = eval_zone_agent.get_agent_state()
    return eval_metrics


class BaseRunner:

    def __init__(
            self,
            run_code: str,
            config: Optional[SingleRunConfig] = None,
            log: Optional[Logger] = None
    ):
        self.initialized = False
        self.run_code: str = run_code
        self.config: SingleRunConfig = config if config is not None else single_run_config
        self.logger: Logger = log if log is not None else logger
        self.run_mode: RunMode = self.config.run.run_mode
        self.use_zones: bool = self.config.environment.zones.enabled
        self.training_steps = self.config.run.training_steps
        self.use_best_evaluation = self.config.run.use_best_evaluation
        self.env: Optional[ZonesEnvWrapper] = None
        self.agent: Optional[AgentAbstract] = None
        self.zone_agent: Optional[AgentAbstract] = None
        self.trucks_wrapper: Optional[TrucksWrapper] = None
        self.is_single_zone: bool = False
        # random_states
        self.random_seed: int = 2
        self.run_random: Optional[RandomState] = None
        # stats tracker
        self.stats_tracker: Tracker = Tracker.init_condensed_tracker(
            run_code=self.run_code, config=self.config, run_mode=self.run_mode, tensorboard=None
        )
        self.run_logger: RunLogger = RunLogger(self.config, self.logger, self.stats_tracker, self.training_steps)
        self.object_handler: Union[ObjectHandler, MinioObjectHandler] = create_object_handler(
            logger=self.logger,
            enabled=self.config.saver.enabled,
            mode=self.config.saver.mode,
            base_path=self.config.saver.get_base_path(),
            default_bucket=self.config.saver.default_bucket,
            minio_endpoint=None
        )
        # eval pool
        self.eval_pool: Optional[Pool] = None
        self.scheduled_eval_counter: int = 0
        self.completed_eval_counter: int = 0
        self.best_eval_performance: Optional[float] = None
        self.best_eval_performance_iteration: int = 0
        self.best_agent_model = None
        self.best_zone_agent_model = None
        self.checkpoint_counter = 1
        self.is_on_policy: bool = True

    @property
    def is_async(self) -> bool:
        return self.config.run.eval_pool_size > 0

    def _init(self):
        if not self.initialized:
            self._init_seeds()
            self._init_env()
            self._init_agent()
            self._init_trucks_wrapper()
            self._init_eval_pool()
            self.initialized = True
            self.run_logger.log_initialized_system(self.random_seed, self.env.n_zones)

    def _init_seeds(self):
        if self.run_mode == RunMode.Eval:
            run_seed = self.config.random_seeds.evaluation[0]
        else:
            run_seed = self.config.random_seeds.training
        self.random_seed = run_seed
        self.run_random: RandomState = RandomState(self.random_seed)

    def _return_new_env(
            self,
            config: SingleRunConfig,
            random_state: RandomState,
            run_code: str,
            run_mode: RunMode,
            random_seed: int,
            log: Logger
    ) -> ZonesEnvWrapper:
        env = ZonesEnvWrapper(
            run_code=run_code,
            config=config,
            random_state=random_state,
            log=log,
            run_mode=run_mode,
            random_seed=random_seed,
        )
        return env

    def _init_eval_env(self, random_seed: int, mode: RunMode = RunMode.Eval, log: Logger = None):
        random_state = RandomState(random_seed)
        eval_config: SingleRunConfig = SingleRunConfig(
            ROOT_DIR,
            **deepcopy(self.config.export(mode=ExportMode.DICT))
        )
        return self._return_new_env(config=eval_config, random_state=random_state,
                                    run_code=f'{self.run_code}_{mode.value}',
                                    run_mode=mode,
                                    random_seed=random_seed,
                                    log=log)

    def _init_env(self):
        self.env: ZonesEnvWrapper = self._return_new_env(config=self.config, random_state=self.run_random,
                                                    run_code=self.run_code, run_mode=self.config.run.run_mode,
                                                    random_seed=self.random_seed, log=self.logger)
        self.is_single_zone = True if len(self.env.zones) == 1 else False

    def _return_new_agent(
            self,
            env: ZonesEnvWrapper,
            random_seed: int,
            init_seed=True,
            run_mode=RunMode.Train,
            log: Logger = None
    ):
        agent = create_agent(
            agent_type=self.config.environment.agent.type,
            action_space=env.action_space,
            random_state=env.random,
            action_spaces=env.action_spaces,
            state_spaces=env.state_spaces,
            nodes=env.nodes,
            zones=env.zones,
            nodes_max_distance=env.nodes_max_distance,
            config=self.config,
            log=self.logger,
            state_builder=env.state_builder,
            is_zone_agent=False,
            mode=run_mode,
            init_seed=init_seed,
            random_seed=random_seed,
            distances_provider=env.distances_provider,
            env=env if self.config.environment.agent.type.is_meta_heuristic() else None,
            **self.config.environment.agent.global_parameters
        )
        if agent.name.is_oracle():
            agent.load_generator = env.load_generator
        return agent

    def _return_new_zone_agent(
            self,
            env: ZonesEnvWrapper,
            random_seed: int,
            init_seed=True,
            run_mode=RunMode.Train,
            log: Logger = None
    ):
        agent = create_agent(
            agent_type=self.config.environment.zones.agent.type,
            action_space=env.zone_action_space,
            random_state=env.random,
            action_spaces=env.action_spaces,
            state_spaces=env.state_spaces,
            nodes=env.nodes,
            zones=env.zones,
            nodes_max_distance=env.nodes_max_distance,
            config=self.config,
            log=log,
            state_builder=env.state_builder,
            is_zone_agent=True,
            mode=run_mode,
            init_seed=init_seed,
            random_seed=random_seed,
            distances_provider=env.distances_provider,
            **self.config.environment.agent.global_parameters
        )
        if agent.name.is_oracle():
            agent.load_generator = env.load_generator
        return agent

    def _init_agent(self):
        self.agent: AgentAbstract = self._return_new_agent(
            self.env, random_seed=self.random_seed, init_seed=True, run_mode=self.run_mode, log=self.logger)
        self.zone_agent: AgentAbstract = self._return_new_zone_agent(
            self.env, random_seed=self.random_seed, init_seed=True, run_mode=self.run_mode, log=self.logger)
        bootstrapping_steps = self.agent.bootstrap_steps
        self.run_logger.log_initialized_agent(self.agent.device, bootstrapping_steps)
        self.run_logger.rollout_size = self.agent.rollout_size
        self._load_agent_model()
        self._load_zone_agent_model()
        self.after_init_agent()
        self.agent.set_mode(self.run_mode)

    def _load_agent_model(self):
        model_load_wrapper = self.config.environment.agent.model_load
        if model_load_wrapper.load_model:
            model_config = model_load_wrapper.load_model_config
            agent_state, model_path = load_agent_model(
                model_load=model_config,
                config_saver_mode=self.config.saver.mode,
                handler=self.object_handler,
                log=self.logger,
                device=self.agent.device,
            )
            self.best_agent_model = agent_state
            self.agent.load_agent_state(agent_state)
            self.run_logger.log_loaded_agent_model(model_path)

    def _load_zone_agent_model(self):
        model_load_wrapper = self.config.environment.zones.agent.model_load
        if model_load_wrapper.load_model:
            model_config = model_load_wrapper.load_model_config
            agent_state, model_path = load_agent_model(
                model_load=model_config,
                config_saver_mode=self.config.saver.mode,
                handler=self.object_handler,
                log=self.logger,
                device=self.zone_agent.device,
            )
            self.best_zone_agent_model = agent_state
            self.zone_agent.load_agent_state(agent_state)
            self.run_logger.log_loaded_agent_model(model_path)

    def _init_trucks_wrapper(self):
        self.trucks_wrapper: TrucksWrapper = TrucksWrapper(
            trucks=self.env.trucks,
            config=self.config
        )

    def _init_eval_pool(self):
        if self.config.run.eval_pool_size > 0:
            set_start_method('spawn', force=True)
            self.eval_pool: Pool = Pool(processes=self.config.run.eval_pool_size)

    def after_init_agent(self):
        self.env.set_stats_tracker(self.stats_tracker)
        self.agent.set_stats_tracker(self.stats_tracker)
        self.zone_agent.set_stats_tracker(self.stats_tracker)

    def run(self):
        self._init()
        if self.agent.name.is_baseline() or self.run_mode != RunMode.Train:
            # no learning involved here, simply run the agent's policy
            self.pre_run_start_callback(training_steps=1, log_train_info=False, log_eval_info=True)
            self.evaluation_run()
        elif self.agent.name.is_on_policy():
            self.is_on_policy = True
            # Collect rollouts and feed them to a policy gradient algorithm
            self.pre_run_start_callback(training_steps=self.training_steps)
            self.train_on_policy()
        elif self.agent.name.is_off_policy():
            self.is_on_policy = False
            # Off-policy algorithm, thus interact and learn directly in the same loop
            self.pre_run_start_callback(training_steps=self.training_steps)
            self.train_off_policy()
        else:
            raise Exception(f'AgentType not valid for Agent: {self.agent.name}')
        self.close_runner()

    def pre_run_start_callback(self, training_steps: int, log_train_info: bool = True, log_eval_info: bool = False):
        if log_train_info:
            self.run_logger.log_training_start(bootstrap_steps=self.agent.bootstrap_steps)
        if log_eval_info:
            self.run_logger.log_evaluation_start()

    def evaluation_run(self):
        self.do_evaluations(self.agent.get_agent_state(), self.zone_agent.get_agent_state())
        self.after_run_callback()

    def do_evaluations(
            self,
            agent_state_model,
            zone_agent_state_model,
            force_no_proc: bool = False,
            eval_counter_forced: int = 0
    ):
        self.run_logger.log_evaluation_start_info(eval_count=self.scheduled_eval_counter)
        if self.eval_pool is not None and not self.agent.name.is_baseline() and not force_no_proc:
            self.eval_pool.apply_async(
                func=_do_evaluations_async,
                kwds={
                    'seeds': self.config.random_seeds.evaluation,
                    'agent_state_model': agent_state_model,
                    'zone_agent_state_model': zone_agent_state_model,
                    'eval_counter': self.scheduled_eval_counter,
                    'config_dict': deepcopy(self.config.export(ExportMode.DICT)),
                    'run_code': self.run_code
                },
                callback=lambda res: self._async_evaluation_callback(res),
                error_callback=lambda e: self.logger.exception(e)
            )
        else:
            counter = self.scheduled_eval_counter if not force_no_proc else eval_counter_forced
            self._do_evaluations(agent_state_model, zone_agent_state_model, counter)
        self.scheduled_eval_counter += 1

    def _async_evaluation_callback(self, result: dict):
        try:
            eval_counter = result['eval_counter']
            eval_agent = result['eval_agent']
            eval_performance = result['eval_performance']
            eval_zone_agent = result['eval_zone_agent']
            del result['eval_counter']
            del result['eval_agent']
            del result['eval_performance']
            del result['eval_zone_agent']
            self.run_logger.log_evaluation_run_async(eval_count=eval_counter + 1, run_mode=self.run_mode, eval_data=result)
            self.after_evaluation_run_callback(
                eval_agent=eval_agent,
                eval_performance=eval_performance,
                eval_counter=eval_counter,
                eval_zone_agent=eval_zone_agent,
                eval_metrics=result,
                is_async=True,
            )
            self.completed_eval_counter += 1
        except Exception as e:
            self.logger.exception(e)
            raise e

    def _do_evaluations(self, agent_state_model, zone_agent_state_model, eval_counter: int, is_async: bool = False):
        seeds = self.config.random_seeds.evaluation
        eval_agent = None
        eval_zone_agent = None
        for i, seed in enumerate(seeds):
            eval_env = self._init_eval_env(random_seed=seed, mode=RunMode.Eval)
            eval_env.set_stats_tracker(self.stats_tracker)
            eval_trucks_wrapper = TrucksWrapper(eval_env.trucks, eval_env.config)
            eval_env.reset(show_log=False)
            eval_zone_agent = self._return_new_zone_agent(
                eval_env, random_seed=seed, init_seed=False, run_mode=RunMode.Eval)
            eval_agent = self._return_new_agent(eval_env, random_seed=seed, init_seed=False, run_mode=RunMode.Eval)
            eval_agent.set_stats_tracker(self.stats_tracker)
            eval_zone_agent.set_stats_tracker(self.stats_tracker)

            if not self.agent.name.is_baseline():
                eval_agent.load_agent_state(agent_state_model)
                eval_zone_agent.load_agent_state(zone_agent_state_model)

            eval_agent.set_mode(RunMode.Eval)
            eval_agent.set_nodes(eval_env.nodes)

            n_episodes = self.config.run.evaluation_episodes
            if eval_env.load_generator.eval_every_week:
                n_episodes = 4

            collect_rollout(
                env=eval_env,
                agent=eval_agent,
                zone_agent=eval_zone_agent,
                trucks_wrapper=eval_trucks_wrapper,
                rollout_size=-1,
                tracker=self.stats_tracker,
                run_mode=RunMode.Eval,
                seed=seed,
                is_bootstrapping=False,
                n_episodes=n_episodes
            )

        track_average_values(tracker=self.stats_tracker, mode=RunMode.Eval, iteration=eval_counter, seeds=seeds)
        if self.config.emulator.model.cdrc_data_model.eval_every_week:
            track_weekly_evaluation(tracker=self.stats_tracker, seeds=seeds, n_weeks=4)
        eval_performance = self.stats_tracker.get(self.config.run.keep_metric)[eval_counter]
        eval_metrics = {'iteration': eval_counter}
        for key in self.stats_tracker.rollout_variables_eval:
            value = self.stats_tracker.get(f'evaluation/avg/{key.replace("step/", "")}')[-1]
            eval_metrics[key.replace('step/', '')] = value
        if is_async:
            eval_metrics['eval_agent'] = eval_agent.get_agent_state()
            eval_metrics['eval_performance'] = eval_performance
            eval_metrics['eval_counter'] = eval_counter
            eval_metrics['eval_zone_agent'] = eval_zone_agent.get_agent_state()
            return eval_metrics
        else:
            self.run_logger.log_evaluation_run_completed(eval_count=eval_counter + 1, run_mode=self.run_mode)
            self.after_evaluation_run_callback(eval_agent.get_agent_state(), eval_performance, eval_counter,
                                               eval_zone_agent.get_agent_state(), eval_metrics)
            self.completed_eval_counter += 1

    def train_on_policy(self):
        self.run_logger.log_training_phase_start()
        # until we reach the stop condition, we collect rollouts of size rollout_size to train the agent
        rollout_size = self.agent.rollout_size
        did_evaluate = False
        for train_i in range(self.training_steps):
            # collect rollout fill the agent's rollout buffer
            collect_rollout(env=self.env, agent=self.agent, zone_agent=self.zone_agent,
                            trucks_wrapper=self.trucks_wrapper,
                            rollout_size=rollout_size, tracker=self.stats_tracker,
                            run_mode=RunMode.Train, seed=None, is_bootstrapping=False)
            # performing a learning step using the current rollout
            if not self.is_single_zone:
                self.zone_agent.learn()
            self.agent.learn()
            # eventually do the evaluation
            did_evaluate = self.training_evaluation(train_iteration=train_i + 1)
            # call the after train step callback
            self.after_train_step_callback(is_on_policy=True, iteration=train_i + 1)
        # training completed, do a final evaluation
        if not did_evaluate:
            # we prevent doing twice the evaluation using the same model if the last step we performed an evaluation
            self.do_evaluations(deepcopy(self.agent.get_agent_state()), deepcopy(self.zone_agent.get_agent_state()))
        # repeat evaluation with the best agent model to save stats
        if not self.is_async:
            self.do_evaluations(self.best_agent_model, self.best_zone_agent_model, force_no_proc=False)
        # eventually save stats and agent model
        self.after_run_callback()

    def train_off_policy(self):
        # bootstrap for bootstrapping steps
        bootstrap_steps = self.agent.bootstrap_steps
        did_evaluate = False
        if bootstrap_steps > 0:
            self.run_logger.log_bootstrapping_start(bootstrap_steps=bootstrap_steps)
            collect_rollout(env=self.env, agent=self.agent, trucks_wrapper=self.trucks_wrapper,
                            rollout_size=bootstrap_steps, tracker=self.stats_tracker,
                            run_mode=RunMode.Train, seed=None, is_bootstrapping=True)
            self.run_logger.log_bootstrapping_end(bootstrap_steps=bootstrap_steps)
        # until we reach the stop condition, we perform steps, we learn and evaluate
        self.run_logger.log_training_phase_start()
        for train_i in range(self.training_steps):
            # use current policy to perform a single step in the environment for each idle truck
            collect_single_step(env=self.env, agent=self.agent, trucks_wrapper=self.trucks_wrapper,
                                tracker=self.stats_tracker, iteration=train_i)
            # perform a learning step
            self.agent.learn()
            # eventually do the evaluation
            did_evaluate = self.training_evaluation(train_iteration=train_i + 1)
            # call the after train step callback
            self.after_train_step_callback(is_on_policy=False, iteration=train_i + 1)
        # training completed, do a final evaluation
        if not did_evaluate:
            # we prevent doing twice the evaluation using the same model if the last step we performed an evaluation
            self.do_evaluations(self.agent.get_agent_state(), self.zone_agent.get_agent_state())
        # repeat evaluation with the best agent model to save stats
        if self.best_eval_performance_iteration != self.scheduled_eval_counter:
            self.do_evaluations(self.best_agent_model, self.best_zone_agent_model)
        # eventually save stats and agent model
        self.after_run_callback()

    def training_evaluation(self, train_iteration: int) -> bool:
        if train_iteration % self.config.run.evaluation_frequency == 0 and train_iteration > 0:
            self.do_evaluations(deepcopy(self.agent.get_agent_state()), deepcopy(self.zone_agent.get_agent_state()))
            return True
        else:
            return False

    def after_train_step_callback(self, is_on_policy: bool, iteration: int):
        if is_on_policy:
            learning_params = {
                **self.zone_agent.get_tracked_learning_params(),
                **self.agent.get_tracked_learning_params()
            }
            self.run_logger.log_on_policy_train_step(iteration=iteration,
                                                     eval_count=self.completed_eval_counter,
                                                     learning_params=learning_params)
        else:
            self.run_logger.log_off_policy_train_step(iteration=iteration,
                                                      eval_count=self.completed_eval_counter,
                                                      learning_params=self.agent.get_tracked_learning_params())

    def after_evaluation_run_callback(
            self,
            eval_agent,
            eval_performance: float,
            eval_counter: int,
            eval_zone_agent,
            eval_metrics: Dict[str, float],
            is_async: bool = False,
    ):
        # save checkpoint
        checkpoint_freq = self.config.saver.checkpoint_frequency
        if checkpoint_freq is not None and eval_counter > 0 and eval_counter % checkpoint_freq == 0:
            self.save_agent_model(eval_agent, filename=f'checkpoint-{self.checkpoint_counter}.pth')
            self.save_agent_model(eval_zone_agent, filename=f'checkpoint-zone-agent-{self.checkpoint_counter}.pth')
            self.checkpoint_counter += 1
        if self.best_eval_performance is None or eval_performance > self.best_eval_performance:
            self.best_eval_performance = eval_performance
            self.best_eval_performance_iteration = eval_counter
            track_best_run_variables(self.stats_tracker, self.best_eval_performance_iteration, eval_metrics, is_async)
            self.best_agent_model = deepcopy(eval_agent)
            self.best_zone_agent_model = deepcopy(eval_zone_agent)
            self.run_logger.log_new_best_evaluation_score(eval_counter + 1, run_mode=self.run_mode)
            self.save_agent_model(self.best_agent_model, filename='best_model.pth')
            self.save_agent_model(self.best_zone_agent_model, filename='best_zone_model.pth')
            return True
        else:
            return False

    def save_agent_model(self, agent_model, filename: str):
        pass

    def after_run_callback(self):
        if self.eval_pool is not None:
            # if eval pool is used, we close it (so no more evaluation can be scheduled)
            # and we wait (join) in case it is still running the latest evaluations
            self.eval_pool.close()
            self.eval_pool.join()
        if self.is_async:
            self.do_evaluations(self.best_agent_model, self.best_zone_agent_model, force_no_proc=True,
                                eval_counter_forced=0)
        run_performance = self.best_eval_performance
        self.save_agent_model(self.agent.get_agent_state(), filename=f'last_model.pth')
        self.save_agent_model(self.zone_agent.get_agent_state(), filename=f'last_zone_model.pth')
        return run_performance

    def close_runner(self):
        if self.run_mode == RunMode.Train:
            if self.agent.name.is_on_policy():
                # log final info
                self.run_logger.log_on_policy_training_completed(self.best_eval_performance_iteration + 1)
            elif self.agent.name.is_off_policy():
                # log final info
                self.run_logger.log_off_policy_training_completed(self.best_eval_performance_iteration + 1)
