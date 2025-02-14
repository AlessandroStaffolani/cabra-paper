from __future__ import print_function

from argparse import ArgumentParser

from cabra.common.command_args import global_parser, check_args


def config_commands(controller_parser):
    parser_sub = controller_parser.add_parser('config', help='Config command')
    sub_subparsers = parser_sub.add_subparsers(dest='action')
    parser_sub_subparsers = sub_subparsers.add_parser('export', help='Export configuration')
    parser_sub_subparsers.add_argument('export_target', help='export target')
    parser_sub_subparsers.add_argument('-m', '--mode', default='yaml', help='Export format. Supported: yaml or json')
    global_parser(parser_sub_subparsers)


def server_commands(controller_parser):
    sub = controller_parser.add_parser('server', help='Server command')
    sub_subparsers = sub.add_subparsers(dest='action')
    parser = sub_subparsers.add_parser('start', help='Start in server mode')
    global_parser(parser)


def script_commands(controller_parser):
    sub = controller_parser.add_parser('script', help='Script mode command')
    sub_subparsers = sub.add_subparsers(dest='action')
    parser = sub_subparsers.add_parser('start', help='Start in script mode')
    parser.add_argument('n', default=1, type=int,
                        help='Number of steps to generate. It overrides the config: "simulation.stop_step"')
    global_parser(parser)


def plot_commands(controller_parser):
    sub = controller_parser.add_parser('plot', help='Plot command')
    sub_subparsers = sub.add_subparsers(dest='action')
    parser = sub_subparsers.add_parser('data', help='Plot the data generated by the emulator')
    parser.add_argument('folder', help='Folder path where to load the data to plot')
    parser.add_argument('-s', '--save', default=False, action='store_true',
                        help='Save the plot images on disk. By default it does not save the plot images')
    parser.add_argument('--save-path', default='data/emulator/plots',
                        help='Path where to save the plot images. Default "data/emulator/plots"')
    parser.add_argument('--save-extension', default='pdf',
                        help='Extension used for saving the plot images. Default "pdf"')
    parser.add_argument('--plot-filters', default='hour,week_day,month,year',
                        help='Comma separated list of plot filters, each filter create a different plot.'
                             ' Default "hour,week_day,month,year"')
    global_parser(parser)


def test_nodes_commands(controller_parser):
    parser_sub = controller_parser.add_parser('test-model', help='Test model command')
    sub_subparsers = parser_sub.add_subparsers(dest='action')
    parser_sub_subparsers = sub_subparsers.add_parser('nodes', help='Generates test nodes')
    global_parser(parser_sub_subparsers)


def add_data_parser(module_parser):
    sub = module_parser.add_parser('data', help='Data loader module')
    controller_parser = sub.add_subparsers(dest='controller')
    test_nodes_commands(controller_parser)
