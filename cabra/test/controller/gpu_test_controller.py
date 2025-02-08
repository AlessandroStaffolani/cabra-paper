import os
import time
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from urllib.parse import quote_plus

import pymongo
import redis
import torch

from cabra import logger

from cabra.common.controller import Controller


def print_proc(index):
    logger.info(f"Hello I'm process {index}")
    time.sleep(2)
    logger.info(f'Process {index} completed execution')


class GPUTestController(Controller):

    def __init__(self):
        super().__init__('GPUTestController')
        self._add_action('start', self.start_gpu_test)
        self.mongo_user = os.getenv('MONGO_USER')
        self.mongo_password = os.getenv('MONGO_PASSWORD')
        self.mongo_host = os.getenv('MONGO_HOST')
        self.mongo_port = int(os.getenv('MONGO_PORT'))
        self.redis_host = os.getenv('REDIS_HOST')
        self.redis_port = int(os.getenv('REDIS_PORT'))
        self.redis_db = int(os.getenv('REDIS_DB'))
        self.redis_password = os.getenv('REDIS_PASSWORD')

    def start_gpu_test(
            self,
            skip_mongo: bool = False,
            skip_redis: bool = False,
            skip_processes: bool = False,
            skip_cuda: bool = False,
            **kwargs):
        if not skip_mongo:
            self.test_mongo()
        if not skip_redis:
            self.test_redis()
        if not skip_processes:
            self.test_processes()
        if not skip_cuda:
            self.test_cuda()

    def test_cuda(self):
        try:
            if not torch.cuda.is_available():
                logger.warning("Torch CUDA acceleration is NOT available!")
            else:
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    logger.info(f"I am GPU number {i}, but you can call me {name}")
        except Exception as e:
            logger.error('Error during cuda test')
            logger.exception(e)

    def test_mongo(self):
        try:
            mongo_uri = f'mongodb://{quote_plus(self.mongo_user)}:{quote_plus(self.mongo_password)}@{self.mongo_host}:{self.mongo_port}'
            mongo = pymongo.MongoClient(mongo_uri, serverSelectionTimeoutMS=10)
            logger.info(mongo.server_info())
        except Exception as e:
            logger.error('Error during mongo test')
            logger.exception(e)

    def test_redis(self):
        try:
            r = redis.Redis(host=self.redis_host, port=self.redis_port, db=self.redis_db, password=self.redis_password)
            logger.info(r.ping())
            logger.info(r.set('test-key', 'test-on-gpu-server'))
            logger.info(r.set('test-key-2', 'test-on-gpu-server'))
            logger.info(r.get('test-key-1'))
            logger.info(r.get('test-key-2'))
            logger.info(r.delete('test-key-2'))
            logger.info(r.delete('test-key-1'))
            logger.info(r.delete('test-key'))
        except Exception as e:
            logger.error('Error during redis test')
            logger.exception(e)

    def test_processes(self):
        try:
            logger.info(f'Processes currently available: {cpu_count()}')
            pool = Pool(processes=cpu_count())

            for i in range(cpu_count()):
                pool.apply_async(func=print_proc, args=(i, ), error_callback=lambda err: logger.exception(err))
            pool.close()
            pool.join()
            logger.info('All processes have executed their test')

        except Exception as e:
            logger.error('Error during processes test')
            logger.exception(e)
