from cabra import ROOT_DIR, logger, __version__
from cabra.global_parser import global_parse_arguments
from cabra.common.controller import get_global_controller
from cabra.common.controller.main_controller import MainController
from cabra.common.command_args import get_main_controller_args
from cabra.emulator.controller import get_main_controller as data_controller
from cabra.run.controller import get_main_controller as run_controller
from cabra.test.controller import get_main_controller as test_controller

MODULE_MAIN_CONTROLLER_MAPPING = {
    'data': data_controller,
    'run': run_controller,
    'global': get_global_controller,
    'test': test_controller,
}

if __name__ == '__main__':
    logger.info(f'Starting system version: {__version__}')
    logger.info(f'Project root dir: {ROOT_DIR}')
    args = global_parse_arguments()
    module, controller, action, action_arguments = get_main_controller_args(args)
    current_controller: MainController = MODULE_MAIN_CONTROLLER_MAPPING[module]()
    current_controller.execute(controller, action, action_arguments)
