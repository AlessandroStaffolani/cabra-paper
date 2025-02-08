from cabra.common.command_args import global_parser


def gpu_tests_commands(controller_parser):
    parser_sub = controller_parser.add_parser('gpu', help='GPU run test')
    sub_subparsers = parser_sub.add_subparsers(dest='action')
    parser_sub_subparsers = sub_subparsers.add_parser('start', help='Start a GPU test run')
    parser_sub_subparsers.add_argument('--skip-mongo', action='store_true', help='Skip mongo test if present')
    parser_sub_subparsers.add_argument('--skip-redis', action='store_true', help='Skip redis test if present')
    parser_sub_subparsers.add_argument('--skip-processes', action='store_true', help='Skip processes test if present')
    parser_sub_subparsers.add_argument('--skip-cuda', action='store_true', help='Skip cuda test if present')
    global_parser(parser_sub_subparsers)


def add_test_parser(module_parser):
    sub = module_parser.add_parser('test', help='Test module')
    controller_parser = sub.add_subparsers(dest='controller')
    gpu_tests_commands(controller_parser)
