import os
import shutil
from logging import Logger
from typing import Optional, Dict, Union, Any, Tuple

import torch

from cabra import SingleRunConfig, get_logger, ROOT_DIR
from cabra.common.filesystem import create_directory
from cabra.common.object_handler import create_object_handler, SaverMode
from cabra.common.object_handler.base_handler import ObjectHandler
from cabra.common.object_handler.minio_handler import MinioObjectHandler
from cabra.environment.config import ModelLoadConfig
from cabra.run.remote import MongoRunWrapper, RedisRunWrapper


def get_run_logger(run_code: str, run_config: SingleRunConfig) -> Logger:
    model_name = run_config.emulator.model.type
    agent_name = run_config.environment.agent.type
    log_folder = os.path.join('bs_repositioning', model_name, agent_name, run_code)
    create_directory(os.path.join(run_config.logs_base_dir, log_folder))
    run_config.environment.logger.update_file_handler_folder(log_folder)
    run_config.environment.logger.name = run_code
    return get_logger(run_config.environment.logger)


def clean_log_folder(run_code: str, run_config: SingleRunConfig):
    model_name = run_config.emulator.model.type
    agent_name = run_config.environment.agent.type
    log_folder = os.path.join(run_config.logs_base_dir, 'bs_repositioning', model_name, agent_name, run_code)
    warning_path = os.path.join(log_folder, 'warning.log')
    if os.path.getsize(warning_path) == 0:
        # no warning or error we can delete the log
        shutil.rmtree(log_folder, ignore_errors=True)
        try:
            shutil.rmtree(log_folder)
        except Exception:
            pass


def complete_training_run(
        run_code: str,
        mongo: MongoRunWrapper,
        config: SingleRunConfig,
        redis: Optional[RedisRunWrapper] = None,
):
    if config.multi_run.is_multi_run:
        is_saved = mongo.is_run_saved_in_multi_run_table(
            run_code=run_code,
            multi_run_code=config.multi_run.multi_run_code
        )
        if not is_saved:
            mongo.save_multi_run_single_run(
                multi_run_code=config.multi_run.multi_run_code,
                run_code=run_code,
                run_params=config.multi_run.multi_run_params
            )
    if redis is not None:
        redis.delete_single_run_stats(run_code=run_code)


def load_agent_model(
        model_load: ModelLoadConfig,
        config_saver_mode: SaverMode,
        handler: Union[ObjectHandler, MinioObjectHandler],
        log: Logger,
        device: torch.device,
) -> Tuple[Dict[str, Any], str]:
    object_handler = handler
    if model_load.mode != config_saver_mode:
        minio_endpoint = None
        object_handler = create_object_handler(
            logger=log,
            enabled=True,
            mode=model_load.mode,
            base_path=model_load.base_path,
            default_bucket=model_load.base_path,
            minio_endpoint=minio_endpoint
        )
    if not object_handler.exists(object_handler.get_path(model_load.path, base_path=model_load.base_path)):
        raise AttributeError(f'Agent model to load path not exists. Rollout "{model_load.path}"')
    agent_state = object_handler.load_agent_model(file_path=model_load.path, map_location=device,
                                                  base_path=model_load.base_path, bucket=model_load.base_path)
    return agent_state, model_load.path
