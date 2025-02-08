from cabra.common.command_args import global_parser


def runner_commands(controller_parser):
    parser_sub = controller_parser.add_parser('runner', help='Runner command')
    sub_subparsers = parser_sub.add_subparsers(dest='action')
    parser_sub_subparsers = sub_subparsers.add_parser('train', help='Train the agent')
    parser_sub_subparsers = sub_subparsers.add_parser('test-runs', help='Test runs')
    parser_sub_subparsers.add_argument('--multi', action='store_true', help='If present it tests a multi run config')
    parser_sub_subparsers.add_argument('--use-mpi', action='store_true', help='If present run a test run with MPI')
    parser_sub_subparsers.add_argument('-p', '--processes', type=int, default=4,
                                       help='If --use-mpi this arguments set the number of parallel MPI workers.'
                                            ' Default 4')
    global_parser(parser_sub_subparsers)


def worker_commands(controller_parser):
    parser_sub = controller_parser.add_parser('worker', help='Worker command')
    sub_subparsers = parser_sub.add_subparsers(dest='action')

    def add_arguments(parser):
        parser.add_argument('-p', '--processes', default=4, type=int,
                            help='Number of sub processes to use for running the runs. Default: 4')
        parser.add_argument('-q', '--queue', help='Specify the queue to use. If not specified the default one is used')
        parser.add_argument('--stop-empty', action='store_true',
                            help='If present it will cause the worker to stop if the queue is empty')
        parser.add_argument('--max-runs', type=int, default=None,
                            help='Max number of runs the worker can perform, if not present no limit is set.')
        parser.add_argument('--use-mpi', action='store_true',
                            help='If present it will cause the worker to run '
                                 'using MPI with --processes parallel workers')
        global_parser(parser)

    parser_sub_subparsers = sub_subparsers.add_parser('offline-worker', help='Start an offline worker')
    parser_sub_subparsers.add_argument('-q', '--queue', default='runs_queue',
                                       help='Specify the queue to use. If not specified the default one is used')
    parser_sub_subparsers.add_argument('--stop-empty', action='store_true',
                                       help='If present it will cause the worker to stop if the queue is empty')
    parser_sub_subparsers.add_argument('--max-runs', type=int, default=None,
                                       help='Max number of runs the worker can perform, if not present no limit is set.')
    parser_sub_subparsers = sub_subparsers.add_parser('run-worker', help='Start a run worker')
    add_arguments(parser_sub_subparsers)


def scheduler_commands(controller_parser):
    parser_sub = controller_parser.add_parser('scheduler', help='Schedule run command')
    sub_subparsers = parser_sub.add_subparsers(dest='action')
    parser_sub_subparsers = sub_subparsers.add_parser('one-run', help='Schedule a single run')
    parser_sub_subparsers.add_argument('-o', '--offline', action='store_true',
                                       help='If present the scheduler will operate offline')
    parser_sub_subparsers.add_argument('--is-eval', action='store_true', help='Schedule the runs as Evaluation runs')
    parser_sub_subparsers.add_argument('-l', '--schedule-locally', action='store_true',
                                       help='Schedule the runs locally in the config folder under the queue sub folder')
    parser_sub_subparsers.add_argument('-q', '--queue',
                                       help='Specify the queue to use. If not specified the default one is used')
    global_parser(parser_sub_subparsers)
    parser_sub_subparsers = sub_subparsers.add_parser('multi-runs', help='Schedule a set of runs')
    parser_sub_subparsers.add_argument('-o', '--offline', action='store_true',
                                       help='If present the scheduler will operate offline')
    parser_sub_subparsers.add_argument('--from-folder',
                                       help='Path where to find multi runs config to schedule. '
                                            'It greps all the files with extension "*.yaml" and "*.yml" in the folder')
    parser_sub_subparsers.add_argument('--is-eval', action='store_true', help='Schedule the runs as Evaluation runs')
    parser_sub_subparsers.add_argument('-l', '--schedule-locally', action='store_true',
                                       help='Schedule the runs locally in the config folder under the queue sub folder')
    parser_sub_subparsers.add_argument('-q', '--queue',
                                       help='Specify the queue to use. If not specified the default one is used')
    global_parser(parser_sub_subparsers)


def mpi_commands(controller_parser):
    parser_sub = controller_parser.add_parser('mpi', help='MPI runner command')
    sub_subparsers = parser_sub.add_subparsers(dest='action')
    parser_sub_subparsers = sub_subparsers.add_parser('start', help='Start a run with MPI')
    parser_sub_subparsers.add_argument('run_code', help='Run code to use for loading the run configuration')
    parser_sub_subparsers.add_argument('--test-run', action='store_true',
                                       help='Run the MPI runner in test mode (no saving will be performed)')
    global_parser(parser_sub_subparsers)


def add_run_parser(module_parser):
    sub = module_parser.add_parser('run', help='Environment module')
    controller_parser = sub.add_subparsers(dest='controller')
    runner_commands(controller_parser)
    worker_commands(controller_parser)
    scheduler_commands(controller_parser)
    mpi_commands(controller_parser)
