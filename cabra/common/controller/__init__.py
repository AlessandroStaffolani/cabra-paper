from typing import Dict

from cabra import logger
from cabra.common.controller.abstract_controller import Controller
from cabra.common.controller.main_controller import MainController
from cabra.common.controller.config_controller import ConfigController


CONTROLLERS: Dict[str, Controller.__class__] = {
    'config': ConfigController,
}
CONTROLLER_ALIASES: Dict[str, str] = {}


def get_global_controller() -> MainController:
    global_controller = MainController(logger)
    for name, controller in CONTROLLERS.items():
        global_controller.add_controller(name, controller)
    for name, target in CONTROLLER_ALIASES.items():
        global_controller.add_alias(name, target)
    return global_controller
