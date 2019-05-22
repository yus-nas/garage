from collections import defaultdict
import multiprocessing as mp
import queue

from dowel import logger
import psutil

from garage.misc.overrides import overrides
from garage.sampler.base import BaseSampler
from garage.sampler.streaming_sampler_worker import run, StreamingWorkerConfig


class StreamingSamplerConfig:
    '''An object for configuring the sampler.

    The intent is that this object, in conjunction with the
    StreamingWorkerConfig should be sufficient to avoid subclassing the
    sampler. Instead of subclassing the sampler for e.g. a specific backend,
    implement a specialized StreamingSamplerConfig and specialized
    StreamingWorkerConfig.
    '''

    def __init__(self, samples_per_iteration, get_env_update,
                 get_policy_update):
        self.samples_per_iteration = samples_per_iteration
        self.get_env_update = get_env_update
        self.get_policy_update = get_policy_update

    def samples_for_itr(self, itr):
        if isinstance(self.samples_per_iteration, int):
            return self.samples_per_iteration
        else:
            return self.samples_per_iteration(itr)


class AlgoConfig(StreamingSamplerConfig):
    '''A StreamingSamplerConfig that just uses an algorithm.

    Assumes that the environment never needs to be updated, the policy can be
    pickled, and the number of trajectories to collect is algo.batch_size.

    :param algo: Should have policy and batch_size fields.
    '''

    def __init__(self, algo):
        self.algo = algo

    def get_env_update(self):
        return None

    def get_policy_update(self):
        return self.algo.policy

    def samples_for_itr(self, itr):
        return self.algo.batch_size


class StreamingSampler(BaseSampler):
    '''A batch Sampler which does as little waiting as possible.

    :param env: An environment
    :param policy: A policy
    :param sampler_config: See the StreamingSamplerConfig documentation.
    :param worker_config: See the StreamingWorkerConfig documentation.
    :param daemonize_workers: Whether to make the worker processes into daemon
    processes. Daemon processes are automatically shut down by the OS when the
    parent process is exits, but this can create zombie processes if the daemon
    process had created any child processes itself. Set this to false if your
    environment or policy spawns child processes.
    '''

    def __init__(self,
                 env,
                 policy,
                 sampler_config,
                 worker_config,
                 daemonize_workers=True):
        self.env = env
        self.policy = policy
        self.max_path_length = worker_config.max_path_length
        self.lifetime_started_samples = 0
        self.n_workers = psutil.cpu_count()
        self.next_iteration = 0
        self.config = sampler_config
        self.daemonize_workers = daemonize_workers
        self.worker_config = worker_config
        self.to_sampler = mp.Queue(2 * self.n_workers)

        # Set in start_worker
        self.workers = None
        self.to_worker = None

    @classmethod
    def from_algo(cls, algo, env=None):
        '''Create an instance of the sampler from an algorithm.

        :param algo: An algorithm
        :param env: An environment. Defaults to algo.env
        :returns StreamingSampler:
        '''
        return cls(
            env=env or algo.env,
            policy=algo.policy,
            sampler_config=AlgoConfig(algo),
            worker_config=StreamingWorkerConfig(algo.max_path_length))

    @overrides
    def start_worker(self):
        self.to_worker = [mp.Queue(1) for _ in range(self.n_workers)]
        self.workers = [
            mp.Process(
                target=run,
                kwargs=dict(
                    worker_number=worker_number,
                    to_worker=self.to_worker[worker_number],
                    to_sampler=self.to_sampler,
                    env=self.env,
                    policy=self.policy,
                    config=self.worker_config,
                )) for worker_number in range(self.n_workers)
        ]
        for w in self.workers:
            w.daemon = self.daemonize_workers
            w.start()

    @overrides
    def obtain_samples(self, itr=None, batch_size=None):
        if itr is None:
            itr = self.next_iteration
        self.next_iteration = itr + 1
        completed_samples = 0
        paths = defaultdict(list)
        streaming = set()
        env_update = self.config.get_env_update()
        policy_update = self.config.get_policy_update()
        if batch_size is not None:
            samples_to_collect = batch_size
        else:
            samples_to_collect = self.config.samples_for_itr(itr)
        while completed_samples < samples_to_collect:
            for worker_num, q in enumerate(self.to_worker):
                if worker_num not in streaming:
                    try:
                        q.put_nowait(('start', (env_update, policy_update,
                                                itr)))
                        streaming.add(worker_num)
                    except queue.Full:
                        pass
            tag, contents = self.to_sampler.get()
            if tag == 'trajectory':
                rollout, i = contents
                if i == itr:
                    for k, v in rollout.items():
                        paths[k].append(v)
                    completed_samples += 1
                    print('.', end='', flush=True)
                else:
                    # Receiving paths from previous iterations is normal.
                    # Potentially, we could gather them here, if an off-policy
                    # method wants them.
                    pass
            else:
                raise ValueError('Unknown tag {} with contents {}'.format(
                    tag, contents))
        for q in self.to_worker:
            try:
                q.put_nowait(('stop', ()))
            except queue.Full:
                logger.log('Worker still busy with prior epoch.')
        return paths

    @overrides
    def shutdown_worker(self):
        for q in self.to_worker:
            # These might cause us to block, but ensure that the workers are
            # closed.
            q.put(('exit', ()))
        for q in self.to_worker:
            q.close()
        self.to_sampler.close()
        for w in self.workers:
            w.join()
