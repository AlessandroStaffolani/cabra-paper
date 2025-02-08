from cabra import single_run_config
from cabra.common.controller import Controller
from cabra.emulator.models.test_data_model import create_nodes_file


class ModelController(Controller):

    def __init__(self):
        super(ModelController, self).__init__('ModelController')
        self._add_action('nodes', self.nodes)

    def nodes(self, **kwargs):
        create_nodes_file(config=single_run_config)
