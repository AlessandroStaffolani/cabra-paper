from typing import Dict

from cabra.common.controller import Controller, MainController
from cabra.environment import logger
from cabra.test.controller.gpu_test_controller import GPUTestController

CONTROLLERS: Dict[str, Controller.__class__] = {
    'gpu': GPUTestController,
}
CONTROLLER_ALIASES: Dict[str, str] = {}


def get_main_controller() -> MainController:
    main_controller = MainController(logger)
    for name, controller in CONTROLLERS.items():
        main_controller.add_controller(name, controller)
    for name, target in CONTROLLER_ALIASES.items():
        main_controller.add_alias(name, target)
    return main_controller
