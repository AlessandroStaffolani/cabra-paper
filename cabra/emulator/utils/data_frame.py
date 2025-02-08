from typing import List

import pandas as pd

from cabra.emulator import logger, emulator_config
from cabra.common.object_handler import create_object_handler


def build_df_from_full_data(folder_path: str) -> pd.DataFrame:
    handler = create_object_handler(
        logger=logger,
        enabled=emulator_config.saver.enabled,
        mode=emulator_config.saver.mode,
        base_path=emulator_config.saver.get_base_path(),
        default_bucket=emulator_config.saver.default_bucket
    )
    files = handler.list_objects_name(folder_path, recursive=True, file_suffix='.json')
    loaded_data: List[dict] = []
    for file_path in files:
        loaded_data += handler.load(file_path)
    columns = ['value', 'resource', 'base_station',
               'second_step', 'second', 'minute', 'hour', 'week_day', 'week', 'month', 'year']
    data = []
    for step_data in loaded_data:
        step = step_data['step']
        for resource, res_values in step_data['data'].items():
            for bs_values in res_values:
                for base_station, value in bs_values.items():
                    data.append([
                        value,
                        resource,
                        base_station,
                        step['second_step'],
                        step['second'],
                        step['minute'],
                        step['hour'],
                        step['week_day'],
                        step['week'],
                        step['month'],
                        step['year'],
                    ])
    return pd.DataFrame(data=data, columns=columns)
