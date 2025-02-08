import os
import subprocess
import sys
from logging import Logger
from typing import Optional

import numpy as np
from mpi4py import MPI


def mpi_fork(
        n: int,
        run_code: str,
        bind_to_core: bool = False,
        script_name: Optional[str] = None,
        exit_on_end: bool = False,
        logger: Optional[Logger] = None,
        is_test_run: bool = False
):
    """
    Launches the current script or the script `script_name` with workers linked by MPI.
    Also, terminates the original process that launched it if `exit_on_end` is True.
    If `logger` is not None it also adds some logging messages before and after running the script.
    """
    if n <= 1:
        return
    if os.getenv("IN_MPI") is None:
        env = os.environ.copy()
        env.update(
            MKL_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            IN_MPI="1"
        )
        args = ["mpirun", "--oversubscribe", "-np", str(n)]
        if bind_to_core:
            args += ["-bind-to", "core"]
        if script_name is not None:
            args += [sys.executable, script_name, 'run', 'mpi', 'start']
            if is_test_run:
                args += ['--test-run', run_code]
            else:
                args += [run_code]
        else:
            args += [sys.executable] + sys.argv
        if logger is not None:
            logger.info(f'Starting parallel execution with {n} workers and args: {args}')
        subprocess.check_call(args, env=env)
        if logger is not None:
            logger.info('Parallel execution completed')
        if exit_on_end:
            sys.exit()


def msg(m, string='', logger: Optional[Logger] = None, level: int = 20):
    if logger is not None:
        logger.log(level, ('Message from %d: %s \t ' % (MPI.COMM_WORLD.Get_rank(), string)) + str(m))
    else:
        print(('Message from %d: %s \t ' % (MPI.COMM_WORLD.Get_rank(), string)) + str(m))


def proc_id() -> int:
    """Get rank of calling process."""
    return MPI.COMM_WORLD.Get_rank()


def allreduce(*args, **kwargs):
    return MPI.COMM_WORLD.Allreduce(*args, **kwargs)


def num_procs() -> int:
    """Count active MPI processes."""
    return MPI.COMM_WORLD.Get_size()


def broadcast(x, root=0):
    MPI.COMM_WORLD.Bcast(x, root=root)


def mpi_op(x, op):
    x, scalar = ([x], True) if np.isscalar(x) else (x, False)
    x = np.asarray(x, dtype=np.float32)
    buff = np.zeros_like(x, dtype=np.float32)
    allreduce(x, buff, op=op)
    return buff[0] if scalar else buff


def mpi_sum(x):
    return mpi_op(x, MPI.SUM)


def mpi_avg(x):
    """Average a scalar or vector over MPI processes."""
    return mpi_sum(x) / num_procs()


def mpi_statistics_scalar(x, with_min_and_max=False):
    """
    Get mean/std and optional min/max of scalar x across MPI processes.

    Args:
        x: An array containing samples of the scalar to produce statistics
            for.

        with_min_and_max (bool): If true, return min and max of x in 
            addition to mean and std.
    """
    x = np.array(x, dtype=np.float32)
    global_sum, global_n = mpi_sum([np.sum(x), len(x)])
    mean = global_sum / global_n

    global_sum_sq = mpi_sum(np.sum((x - mean) ** 2))
    std = np.sqrt(global_sum_sq / global_n)  # compute global std

    if with_min_and_max:
        global_min = mpi_op(np.min(x) if len(x) > 0 else np.inf, op=MPI.MIN)
        global_max = mpi_op(np.max(x) if len(x) > 0 else -np.inf, op=MPI.MAX)
        return mean, std, global_min, global_max
    return mean, std
