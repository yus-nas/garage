import os
import queue
import random

import numpy as np


def never_render(worker_number, rollout_number, frame_number):
    '''A `should_render` function which always returns false.

    This is the default value for the `StreamingSampler`'s `should_render`
    parameter.
    :param worker_number: The number of the worker (generally, ranges from 0 to
    the number of CPU's).
    :param rollout_number: Which rollout the worker is currently executing.
    Starts at 0 and increments by 1 every time a rollout is completed by that
    worker.
    :param frame_number: Which frame the worker is currently executing. Starts
    at 0 at the beginning of every rollout, then is incremented by 1 every
    frame.
    :returns: Whether to render this frame.
    '''
    return False


def return_update(obj, update):
    '''An `update_function` which just returns the update.

    Use this if your update parameter is just the object itself.
    :param obj: The object to be updated.
    :param update: The update to be applied.
    :returns: The update.
    '''
    return update


def skip_update(obj, update):
    '''An `update_function` which just returns the object.

    Use this if your update parameter is irrelevant.
    :param obj: The object to be updated.
    :param update: The update to be applied.
    :returns: The "updated" object.
    '''
    return update


def default_worker_start_function(worker_number):
    '''A reasonable worker start function.

    A worker start function which attempts to set the worker process title, and
    reseeds the worker using urandom.
    :param worker_number: An integer between 0 and the number of CPUs.
    '''
    try:
        from setproctitle import setproctitle, getproctitle
        setproctitle('worker({}):{}'.format(worker_number, getproctitle()))
    except ImportError:
        pass
    seed = worker_number + int.from_bytes(os.urandom(4), 'little')
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def rollout(env, policy, max_path_length, worker_number, sample_number,
            should_render):
    '''The rollout function copied from rlkit, and modified to use a callback
    to check if it should render each frame.

    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
    - observations
    - actions
    - rewards
    - next_observations
    - terminals

    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
    - policy_infos
    - env_infos

    :param env:
    :param policy:
    :param max_path_length:
    :param worker_number: The number of the worker (generally, ranges from 0 to
    the number of CPU's).
    :param sample_number: Which rollout the worker is currently executing.
    Starts at 0 and increments by 1 every time a rollout is completed by that
    worker.
    :param should_render: A function which takes worker_number, sampler_number,
    the frame number, and returns a boolean controlling whether this frame is
    rendered. See `never_render()` for more information.
    :return:
    '''
    observations = []
    actions = []
    rewards = []
    terminals = []
    policy_infos = []
    env_infos = []
    o = env.reset()
    policy.reset()
    next_o = None
    path_length = 0
    if should_render(worker_number, sample_number, path_length):
        env.render()
    while path_length < max_path_length:
        a, policy_info = policy.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)
        policy_infos.append(policy_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if should_render(worker_number, sample_number, path_length):
            env.render()

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack((observations[1:, :],
                                   np.expand_dims(next_o, 0)))
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        policy_infos=policy_infos,
        env_infos=env_infos,
    )


class StreamingWorkerConfig:
    '''A configuration object which allows specializing the worker.

    Intended to replace subclassing the StreamingSampler.
    See the documentation for StreamingSamplerConfig for more details.
    '''

    def __init__(self,
                 max_path_length,
                 rollout_function=rollout,
                 worker_start_function=default_worker_start_function,
                 env_update_function=skip_update,
                 policy_update_function=return_update,
                 should_render=never_render):
        self.worker_start_function = worker_start_function
        self.rollout_function = rollout_function
        self.policy_update_function = policy_update_function
        self.env_update_function = env_update_function
        self.should_render = should_render


def run(worker_number, to_worker, to_sampler, env, policy, config):
    '''The streaming worker state machine.

    Starts in the "not streaming" state.
    Enters the "streaming" state when the "start" message is received.
    While in the "streaming" state, it streams rollouts back to the manager.
    When it receives a "stop" message, or the queue back to the manager is
    full, it enters the "not streaming" state.
    When it receives the "exit" message, it terminates.
    '''

    config.worker_start_function()
    iteration = -1
    completed_samples = 0
    streaming_samples = False

    while True:
        if streaming_samples:
            # We're streaming, so try to get a message without waiting. If we
            # can't, the message is "continue", without any contents.
            try:
                tag, contents = to_worker.get_nowait()
            except queue.Empty:
                tag = 'continue'
                contents = None
        else:
            # We're not streaming anymore, so wait for a message.
            tag, contents = to_worker.get()

        if tag == 'start':
            # Update env and policy.
            env_update, policy_update, iteration = contents
            env = config.env_update_function(env, env_update)
            policy = config.policy_update_function(policy, policy_update)
            streaming_samples = True
        elif tag == 'stop':
            streaming_samples = False
        elif tag == 'continue':
            trajectory = config.rollout_function(
                env, policy, config.max_path_length, worker_number,
                completed_samples, config.should_render)
            completed_samples += 1
            try:
                to_sampler.put_nowait(('trajectory', (trajectory, iteration)))
            except queue.Full:
                # Either the sampler has fallen far behind the workers, or we
                # missed a "stop" message. Either way, stop streaming.
                streaming_samples = False
        elif tag == 'exit':
            return
        else:
            raise ValueError('Unknown tag {} with contents {}'.format(
                tag, contents))
