import datetime
from typing import List, Dict, Optional, Tuple

import numpy as np

from cabra import SingleRunConfig
from cabra.common.data_structure import RunMode, Done
from cabra.common.stats_tracker import Tracker
from cabra.environment.action_space import ZoneAction
from cabra.environment.agent.abstract import AgentAbstract
from cabra.environment.agent.experience_replay.experience_entry import RawTransition
from cabra.environment.node import Node
from cabra.environment.state_wrapper import StateWrapper
from cabra.environment.truck import TrucksWrapper
from cabra.environment.wrapper import EnvWrapper, PartialTransition
from cabra.environment.zones_env_wrapper import ZonesEnvWrapper

np.seterr('raise')


def init_nodes_stats_rollout(config: SingleRunConfig, nodes: List[Node]) -> Optional[Dict[str, List[float]]]:
    if config.run.run_mode == RunMode.Eval and not config.saver.stats_condensed:
        nodes_stats: Dict[str, List[float]] = {}
        for i, node in enumerate(nodes):
            nodes_stats[f'nodes/n_{i}/bikes'] = []
            nodes_stats[f'nodes/n_{i}/empty_slots'] = []
        return nodes_stats
    return None


def store_nodes_step_data(nodes_stats: Optional[Dict[str, List[float]]], nodes: List[Node]):
    if nodes_stats is not None:
        for i, node in enumerate(nodes):
            nodes_stats[f'nodes/n_{i}/bikes'].append(node.bikes)
            nodes_stats[f'nodes/n_{i}/empty_slots'].append(node.empty_slots)


def perform_single_step(
        empty_steps: int,
        steps_counter: int,
        rollout_steps: int,
        env: ZonesEnvWrapper,
        agent: AgentAbstract,
        zone_agent: AgentAbstract,
        trucks_wrapper: TrucksWrapper,
        run_mode: RunMode,
        tracker: Tracker,
        running_rollout_prefix: str,
        running_episode_prefix: str,
        is_on_policy: bool = True,
        last_zone_s_w=None,
        last_s_w=None,
) -> Tuple[Done, int, int, int, Optional[StateWrapper], Optional[StateWrapper]]:
    step = env.current_time_step
    is_single_zone = True if len(env.zones) == 1 else False
    default_zone_action = ZoneAction(zone=0)
    step_metrics = {key: 0 for key in tracker.rollout_variables_eval_step}
    if trucks_wrapper.are_trucks_idle(step):
        # get zone state
        if not is_single_zone:
            zone_state_wrapper = env.get_zone_state()
        partial_transitions: List[PartialTransition] = []
        previous_truck = None
        previous_was_wait = False
        previous_reposition_action = None
        for i, truck in enumerate(trucks_wrapper.idle_trucks(step)):
            # unmask all zone actions
            env.unmask_actions()
            # set current truck features and update state with respect to the new values in the previous truck
            if not is_single_zone:
                env.zone_update_current_truck(
                    truck, zone_state_wrapper, previous_truck, previous_was_wait, previous_reposition_action)
                # get zone action
                zone_action, zone_action_info = zone_agent.choose(zone_state_wrapper, step, truck)
            else:
                zone_action = default_zone_action
                zone_action_info = {}
            # we get the action even if the zone_action is a wait action.
            # In case is wait the reposition action space will all be disabled except for the wait action
            reposition_state_wrapper = env.get_reposition_state(zone_action, truck.index)
            # disable unavailable actions
            env.reposition_disable_unavailable_actions(zone_action, truck)
            # get reposition action
            zone_wait, current_zone_id = env.zone_action_values(zone_action)
            reposition_action, reposition_action_info = agent.choose(
                reposition_state_wrapper, step, truck, str(current_zone_id) if not zone_wait else None)
            # store the previous state and action
            env.store_previous_feature_values(truck, reposition_action, zone_action)
            # get state copies before applying action
            if not is_single_zone:
                z_state_w_copy = zone_state_wrapper.copy()
            else:
                z_state_w_copy = None
            state_w_copy = reposition_state_wrapper.copy()
            # apply action
            action_cost, n_shortages, penalty = env.apply_reposition_action(
                reposition_action, truck, str(current_zone_id))
            # store the partial transition, make a copy of the state because we are going to modify again the state
            partial_transitions.append(PartialTransition(
                zone_state_wrapper=z_state_w_copy,
                state_wrapper=state_w_copy,
                zone_action=zone_action,
                action=reposition_action if not env.use_continuous else reposition_action_info['original_action'],
                truck=truck,
                action_cost=action_cost,
                action_shortages=n_shortages,
                penalty=penalty,
                zone_value=zone_action_info['value'] if 'value' in zone_action_info else None,
                zone_log_probs=zone_action_info['log_probs'] if 'log_probs' in zone_action_info else None,
                zone_action_mask=zone_action_info['action_mask'] if 'action_mask' in zone_action_info else None,
                value=reposition_action_info['value'] if 'value' in reposition_action_info else None,
                log_probs=reposition_action_info['log_probs'] if 'log_probs' in reposition_action_info else None,
                action_mask=reposition_action_info['action_mask'] if 'action_mask' in reposition_action_info else None,
                policy_index=reposition_action_info[
                    'policy_index'] if 'policy_index' in reposition_action_info else None,
            ))
            previous_truck = truck
            previous_was_wait, _, _ = env.reposition_action_values(reposition_action)
            previous_reposition_action = reposition_action
            if run_mode == RunMode.Train:
                if agent.is_buffer_full(offset=i + 1) or zone_agent.is_buffer_full(offset=i + 1):
                    # if the rollout buffer is full, we force to stop the rollout
                    break
        # go to the next step, update the demand, get the next_state and the reward with respect to the new demand
        completed_transitions, done = env.step(partial_transitions)
        for entry in completed_transitions:
            if run_mode == RunMode.Train:
                zone_wait, current_zone_id = env.zone_action_values(entry.zone_action)
                agent.push_experience(
                    state_wrapper=entry.state_wrapper, action=entry.action,
                    next_state_wrapper=entry.next_state_wrapper,
                    reward=entry.reward, done=entry.done, value=entry.value, log_probs=entry.log_probs,
                    action_mask=entry.action_mask, zone_index=entry.policy_index
                )
                if not is_single_zone:
                    zone_agent.push_experience(
                        state_wrapper=entry.zone_state_wrapper, action=entry.zone_action,
                        next_state_wrapper=entry.zone_next_state_wrapper,
                        reward=entry.reward, done=entry.done, value=entry.zone_value, log_probs=entry.zone_log_probs,
                        action_mask=entry.zone_action_mask
                    )
            # update stats
            track_step_wrapper(tracker, entry, steps_counter, done, empty_steps,
                               is_eval=run_mode == RunMode.Eval, is_on_policy=is_on_policy,
                               running_episode_prefix=running_episode_prefix,
                               running_rollout_prefix=running_rollout_prefix,
                               step_metrics=step_metrics)
            empty_steps = 0
            steps_counter = 0
            rollout_steps += 1
            last_zone_s_w = entry.zone_next_state_wrapper
            last_s_w = entry.next_state_wrapper
    else:
        done = env.empty_step()
        empty_steps += 1
    if run_mode == RunMode.Eval:
        tracker.track_eval_step_metrics(prefix=running_rollout_prefix, step_metrics=step_metrics)
    return done, empty_steps, steps_counter, rollout_steps, last_zone_s_w, last_s_w


def collect_rollout(
        env: ZonesEnvWrapper,
        agent: AgentAbstract,
        zone_agent: AgentAbstract,
        trucks_wrapper: TrucksWrapper,
        rollout_size: int,
        tracker: Tracker,
        run_mode: RunMode = RunMode.Train,
        seed: Optional[int] = None,
        is_bootstrapping: bool = False,
        n_episodes: Optional[int] = None
):
    is_single_zone = True if len(env.zones) == 1 else False
    rollout_prefix, eval_array_prefix, eval_history_prefix = '', '', ''
    if run_mode == RunMode.Train:
        running_rollout_prefix = f'{run_mode.value}/rollout_running/'
        rollout_prefix = f'{run_mode.value}/avg-1-steps/'
    else:
        running_rollout_prefix = f'{run_mode.value}-{seed}/rollout_running/'
        eval_array_prefix = f'{run_mode.value}-{seed}/'
        eval_history_prefix = f'{run_mode.value}-{seed}/history/'
    running_episode_prefix = f'{run_mode.value}/episode_running/'
    episode_prefix = f'{run_mode.value}/episode/'

    last_value, last_done, last_zone_s_w, last_s_w = 0, False, None, None

    if not env.ready:
        env.reset(show_log=False)

    empty_steps: int = 0
    episodes = 0
    steps_counter = 1

    rollout_i = 0
    agent.start_rollout()
    zone_agent.start_rollout()
    while not rollout_stop_condition_reached(rollout_i, rollout_size, episodes, n_episodes, agent.continue_training,
                                             run_mode):
        done, empty_steps, steps_counter, rollout_i, last_zone_s_w, last_s_w = perform_single_step(
            empty_steps=empty_steps, steps_counter=steps_counter, rollout_steps=rollout_i,
            env=env, agent=agent, zone_agent=zone_agent, trucks_wrapper=trucks_wrapper, run_mode=run_mode,
            tracker=tracker, running_rollout_prefix=running_rollout_prefix,
            running_episode_prefix=running_episode_prefix, is_on_policy=True,
            last_zone_s_w=last_zone_s_w, last_s_w=last_s_w
        )

        steps_counter += 1

        if agent.name.is_meta_heuristic():
            n_steps = int(rollout_i / len(env.trucks))
            total_steps = len(env.load_generator.dataset)
            prefix = f'{RunMode.Eval}-{seed}/history'
            avg_solution_time = tracker.get_key_to_string(f'{prefix}/single_truck_action_time', np.mean)
            avg_solution_time_raw_value = np.mean(tracker.get(f'{prefix}/single_truck_action_time'))
            expected_run_time = avg_solution_time_raw_value * len(env.trucks) * (total_steps - n_steps)
            env.logger.info(f'{n_steps}/{total_steps} evaluation steps performed by {agent.name} agent | '
                            f'avg single truck action time = {avg_solution_time} seconds | '
                            f'Expected remaining time {datetime.timedelta(seconds=expected_run_time)}')

        if done.to_bool():
            _ = env.reset(done)
            episodes += 1
            if run_mode == RunMode.Train:
                tracker.track_episode_end(running_episode_prefix, episode_prefix)

    if run_mode == RunMode.Train:
        if not is_bootstrapping:
            if not is_single_zone:
                last_zone_value = zone_agent.evaluate_state_value(last_zone_s_w)
                zone_agent.end_rollout(last_zone_value, last_done)
            last_value = agent.evaluate_state_value(last_s_w)
            agent.end_rollout(last_value, last_done)
            tracker.track_rollout_end(running_rollout_prefix, train_prefix=rollout_prefix, is_eval=False)
            tracker.track(f'{run_mode.value}/episodes', episodes)
    else:
        tracker.track_rollout_end(running_rollout_prefix,
                                  eval_array_prefix=eval_array_prefix,
                                  eval_history_prefix=eval_history_prefix,
                                  is_eval=True)


def rollout_stop_condition_reached(
        iteration: int,
        rollout_size: int,
        episodes: int,
        n_episodes: Optional[int] = None,
        continue_training: bool = True,
        run_mode: RunMode = RunMode.Train,
) -> bool:
    if run_mode == RunMode.Train and continue_training is False:
        return True
    if n_episodes is not None:
        return episodes >= n_episodes
    else:
        return iteration >= rollout_size


def collect_single_step(
        env: EnvWrapper,
        agent: AgentAbstract,
        trucks_wrapper: TrucksWrapper,
        tracker: Tracker,
        iteration: int,
) -> int:
    step_actions = 0
    if not env.ready:
        env.reset(show_log=False)
    step = env.current_time_step
    if trucks_wrapper.are_trucks_idle(step):
        # get current step state
        state_wrapper = env.get_state()
        # we store partial transition here
        partial_transitions: List[PartialTransition] = []
        for truck in trucks_wrapper.idle_trucks(step):
            # enable all actions for safety
            env.unmask_actions()
            # set the truck in the state
            truck_state_wrapper = env.update_truck_features(current_truck=truck, state_wrapper=state_wrapper)
            # disable truck unavailable actions
            env.disable_truck_unavailable_actions(truck)
            # choose action
            action, action_info = agent.choose(truck_state_wrapper, step, truck)
            # store the previous state and action
            env.store_previous_feature_values(action, truck_state_wrapper, truck)
            # copy the state, before applying the action on it, for the partial transition
            state_wrapper_copy = truck_state_wrapper.copy()
            # apply the action
            action_cost, n_shortages, penalty = env.apply_action(action, truck)
            # store the partial transition, make a copy of the state because we are going to modify again the state
            partial_transitions.append(PartialTransition(
                state_wrapper=state_wrapper_copy,
                action=action,
                truck=truck,
                action_cost=action_cost,
                action_shortages=n_shortages,
                penalty=penalty,
                value=action_info['value'] if 'value' in action_info else None,
                log_probs=action_info['log_probs'] if 'log_probs' in action_info else None,
                action_mask=action_info['action_mask'] if 'action_mask' in action_info else None
            ))

        # go to the next step, update the demand, get the next_state and the reward with respect to the new demand
        completed_transitions, done = env.step(partial_transitions)
        for entry in completed_transitions:
            # push transition in the experience replay
            agent.push_experience(
                state_wrapper=entry.state_wrapper, action=entry.action,
                next_state_wrapper=entry.next_state_wrapper,
                reward=entry.reward, done=entry.done, value=entry.value, log_probs=entry.log_probs,
                action_mask=entry.action_mask
            )
            # update stats
            track_off_policy_training_iteration(tracker=tracker, reward=entry.reward,
                                                step_info=entry.step_info, done=done.to_bool())
            step_actions += 1

    else:
        done = env.empty_step()
    if done.to_bool():
        _ = env.reset(done)
    return step_actions


def track_step_wrapper(
        tracker: Tracker,
        raw_transition: RawTransition,
        total_steps: int,
        done: Optional[Done],
        empty_steps: int = 0,
        is_eval: bool = False,
        is_on_policy: bool = True,
        running_rollout_prefix: str = '',
        running_episode_prefix: str = '',
        step_metrics: Dict[str, float] = {},
        **kwargs
):
    if is_on_policy:
        tracker.track_running_episode_and_rollout_variables(
            rollout_prefix=running_rollout_prefix, episode_prefix=running_episode_prefix,
            reward=raw_transition.reward,
            shortages=raw_transition.step_info['reward_info']['shortages'],
            env_shortages=raw_transition.step_info['reward_info']['env_shortages'],
            cost=raw_transition.step_info['reward_info']['cost'],
            solved_steps=1 if raw_transition.step_info['solved_step'] is True else 0,
            penalties=raw_transition.step_info['reward_info']['penalty'],
            empty_steps=empty_steps,
            total_steps=total_steps,
            is_eval=is_eval
        )
        if is_eval:
            step_metrics['step/reward'] += raw_transition.reward
            step_metrics['step/shortages'] += raw_transition.step_info['reward_info']['shortages']
            step_metrics['step/env_shortages'] += raw_transition.step_info['reward_info']['env_shortages']
            step_metrics['step/cost'] += raw_transition.step_info['reward_info']['cost']
            step_metrics['step/solved_steps'] += 1 if raw_transition.step_info['solved_step'] is True else 0
    else:
        track_off_policy_training_iteration(
            tracker=tracker, reward=raw_transition.reward, step_info=raw_transition.step_info, done=done.to_bool())


def _get_single_average_value(tracker: Tracker, prefix, key: str, iteration: int, seeds: List[int]) -> float:
    values = []
    for seed in seeds:
        values.append(tracker.get(f'{prefix}-{seed}/{key}')[iteration])
    return np.mean(values).item()


def _get_single_average_value_cumulative_only(
        tracker: Tracker,
        prefix, key: str,
        seeds: List[int]
) -> float:
    values = []
    for seed in seeds:
        values.append(tracker.get(f'{prefix}-{seed}/{key}'))
    return np.mean(values).item()


def track_average_values(
        tracker: Tracker,
        mode: RunMode,
        iteration: int,
        seeds: List[int]
):
    prefix = mode.value
    for metric in tracker.rollout_variables_eval:
        tracker.track(
            f'{prefix}/avg/{metric}',
            _get_single_average_value(tracker, prefix, f'{metric}/total', iteration, seeds),
            iteration,
        )


def track_weekly_evaluation(
        tracker: Tracker,
        seeds: List[int],
        n_weeks: int
):
    metrics = ['cost', 'env_shortages', 'reward', 'shortages', 'solved_steps']
    for seed in seeds:
        for metric in metrics:
            key = f'evaluation-{seed}/history/step/{metric}'
            data = tracker.get(key)
            week_len = len(data) // n_weeks
            weeks_data: Dict[int, List[float]] = {week: [] for week in range(n_weeks)}
            for week in range(n_weeks):
                for index in range(week * week_len, week * week_len + week_len):
                    weeks_data[week].append(data[index])
            for week, week_data in weeks_data.items():
                new_key = f'evaluation-{seed}/week-{week}-history/step/{metric}'
                tracker.init_tracking(new_key, disk_save=True)
                tracker.track(new_key, week_data, replace_value=True)


def track_best_run_variables(
        tracker: Tracker,
        best_iteration: int,
        eval_metrics: dict,
        is_async: bool = False
):
    avg_prefix = f'{RunMode.Eval}/avg'
    best_run_prefix = f'{RunMode.Eval}/best_run'
    if is_async:
        for key, value in eval_metrics.items():
            tracker.track(f'{best_run_prefix}/{key}', value)
    else:
        for var in tracker.rollout_variables_eval:
            value = tracker.get(f'{avg_prefix}/{var}')[best_iteration]
            tracker.track(f'{best_run_prefix}/{var}', value)


def track_off_policy_training_iteration(
        tracker: Tracker,
        reward: float,
        step_info: dict,
        done: bool,
):
    prefix = RunMode.Train.value
    metrics_window = tracker.config.run.metrics_window
    tracker.track(f'{prefix}/reward', reward)
    tracker.track(f'{prefix}/cost', step_info['reward_info']['cost'])
    tracker.track(f'{prefix}/shortages', step_info['reward_info']['shortages'])
    tracker.track(f'{prefix}/env_shortages', step_info['reward_info']['env_shortages'])
    tracker.track(f'{prefix}/solved_steps', 1 if step_info['solved_step'] is True else 0)
    tracker.track(f'{prefix}/penalties', step_info['reward_info']['penalty'])
    if done:
        tracker.track(f'{prefix}/episodes', 1)

    tracker.track(f'{prefix}/avg-{metrics_window}-steps/total_steps', 1)
    tracker.track(f'{prefix}/avg-{metrics_window}-steps/reward', reward)
    tracker.track(f'{prefix}/avg-{metrics_window}-steps/cost', step_info['reward_info']['cost'])
    tracker.track(f'{prefix}/avg-{metrics_window}-steps/shortages', step_info['reward_info']['shortages'])
    tracker.track(f'{prefix}/avg-{metrics_window}-steps/env_shortages', step_info['reward_info']['env_shortages'])
    tracker.track(f'{prefix}/avg-{metrics_window}-steps/solved_steps', 1 if step_info['solved_step'] is True else 0)
    tracker.track(f'{prefix}/avg-{metrics_window}-steps/penalties', step_info['reward_info']['penalty'])


def append_not_none(array: list, value):
    if value is not None:
        array.append(value)

# def collect_rollout(
#         env: EnvWrapper,
#         agent: AgentAbstract,
#         zone_agent: AgentAbstract,
#         trucks_wrapper: TrucksWrapper,
#         rollout_size: int,
#         tracker: Tracker,
#         iteration: int,
#         run_logger: RunLogger,
#         run_mode: RunMode = RunMode.Train,
#         seed: Optional[int] = None,
#         is_bootstrapping: bool = False,
#         n_episodes: Optional[int] = None
# ):
#     rollout_prefix, eval_array_prefix, eval_history_prefix = '', '', ''
#     if run_mode == RunMode.Train:
#         running_rollout_prefix = f'{run_mode.value}/rollout_running/'
#         rollout_prefix = f'{run_mode.value}/avg-1-steps/'
#     else:
#         running_rollout_prefix = f'{run_mode.value}-{seed}/rollout_running/'
#         eval_array_prefix = f'{run_mode.value}-{seed}/'
#         eval_history_prefix = f'{run_mode.value}-{seed}/history/'
#     running_episode_prefix = f'{run_mode.value}/episode_running/'
#     episode_prefix = f'{run_mode.value}/episode/'
#
#     last_value, last_done, last_state_wrapper = 0, False, None
#
#     if not env.ready:
#         env.reset(show_log=False)
#
#     empty_steps: int = 0
#     episodes = 0
#     steps_counter = 1
#
#     rollout_i = 0
#     agent.start_rollout()
#     while not rollout_stop_condition_reached(rollout_i, rollout_size, episodes, n_episodes, agent.continue_training,
#                                              run_mode):
#         step = env.current_time_step
#         if trucks_wrapper.are_trucks_idle(step):
#             # get current step state
#             state_wrapper = env.get_state()
#             # we store partial transition here
#             partial_transitions: List[PartialTransition] = []
#             for i, truck in enumerate(trucks_wrapper.idle_trucks(step)):
#                 # enable all actions for safety
#                 env.unmask_actions()
#                 # set the truck in the state
#                 truck_state_wrapper = env.update_truck_features(current_truck=truck, state_wrapper=state_wrapper)
#                 # disable truck unavailable actions
#                 env.disable_truck_unavailable_actions(truck)
#                 # choose action
#                 action, action_info = agent.choose(truck_state_wrapper, step, truck)
#                 # store the previous state and action
#                 env.store_previous_feature_values(action, truck_state_wrapper, truck)
#                 # copy the state, before applying the action on it, for the partial transition
#                 state_wrapper_copy = truck_state_wrapper.copy()
#                 # apply the action
#                 action_cost, n_shortages, penalty = env.apply_action(action, truck)
#                 # store the partial transition, make a copy of the state because we are going to modify again the state
#                 partial_transitions.append(PartialTransition(
#                     state_wrapper=state_wrapper_copy,
#                     action=action if not env.use_continuous else action_info['original_action'],
#                     truck=truck,
#                     action_cost=action_cost,
#                     action_shortages=n_shortages,
#                     penalty=penalty,
#                     value=action_info['value'] if 'value' in action_info else None,
#                     log_probs=action_info['log_probs'] if 'log_probs' in action_info else None,
#                     action_mask=action_info['action_mask'] if 'action_mask' in action_info else None
#                 ))
#                 if run_mode == RunMode.Train:
#                     if agent.is_buffer_full(offset=i + 1):
#                         # if the rollout buffer is full, we force to stop the rollout
#                         break
#
#             # go to the next step, update the demand, get the next_state and the reward with respect to the new demand
#             completed_transitions, done = env.step(partial_transitions)
#             for entry in completed_transitions:
#                 if run_mode == RunMode.Train:
#                     agent.push_experience(
#                         state_wrapper=entry.state_wrapper, action=entry.action,
#                         next_state_wrapper=entry.next_state_wrapper,
#                         reward=entry.reward, done=entry.done, value=entry.value, log_probs=entry.log_probs,
#                         action_mask=entry.action_mask
#                     )
#                 # update stats
#                 tracker.track_running_episode_and_rollout_variables(
#                     rollout_prefix=running_rollout_prefix, episode_prefix=running_episode_prefix,
#                     reward=entry.reward,
#                     shortages=entry.step_info['reward_info']['shortages'],
#                     env_shortages=entry.step_info['reward_info']['env_shortages'],
#                     cost=entry.step_info['reward_info']['cost'],
#                     solved_steps=1 if entry.step_info['solved_step'] is True else 0,
#                     penalties=entry.step_info['reward_info']['penalty'],
#                     empty_steps=empty_steps,
#                     total_steps=steps_counter,
#                     is_eval=run_mode == RunMode.Eval
#                 )
#                 empty_steps = 0
#                 steps_counter = 0
#
#                 if is_bootstrapping:
#                     run_logger.log_bootstrapping_status(rollout_i, rollout_size)
#
#                 last_done = done
#                 last_state_wrapper = entry.next_state_wrapper
#                 # we increase the rollout step after each non-empty step
#                 rollout_i += 1
#         else:
#             done = env.empty_step()
#             empty_steps += 1
#         steps_counter += 1
#
#         if done.to_bool():
#             _ = env.reset(done)
#             episodes += 1
#             if run_mode == RunMode.Train:
#                 tracker.track_episode_end(running_episode_prefix, episode_prefix)
#
#     if run_mode == RunMode.Train:
#         if not is_bootstrapping:
#             last_value = agent.evaluate_state_value(last_state_wrapper)
#             agent.end_rollout(last_value, last_done)
#             tracker.track_rollout_end(running_rollout_prefix, train_prefix=rollout_prefix, is_eval=False)
#             tracker.track(f'{run_mode.value}/episodes', episodes)
#     else:
#         tracker.track_rollout_end(running_rollout_prefix,
#                                   eval_array_prefix=eval_array_prefix,
#                                   eval_history_prefix=eval_history_prefix,
#                                   is_eval=True)
