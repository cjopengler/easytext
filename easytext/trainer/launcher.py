#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
训练启动器

Authors: PanXu
Date:    2020/12/10 10:20:00
"""

from typing import List, Union, Optional
import logging

import torch
from torch import distributed as TorchDist

from easytext.trainer import Config
from easytext.trainer import DistributedParameter


class Launcher:
    """
    训练启动器
    """

    def __init__(self, config: Config = None):
        """
        启动器初始化
        :param config: 当前的配置
        """
        self.config = config
        self._devices = self._init_devices()
        self._distributed_parameter = self._init_distributed_parameter()

    def _init_devices(self) -> Union[List[torch.device]]:
        """
        返回训练需要用到的设备
        :return: device list, 如果是单一设备，返回的也是 List.
        """
        raise NotImplementedError()

    def _init_distributed_parameter(self) -> Optional[DistributedParameter]:
        """
        初始化 多GPU/分布式参数
        :return:
        """
        raise NotImplementedError

    def _start_process(self, rank: Optional[int]) -> None:
        """
        启动训练过程的函数,子类中实现完整的训练过程。一般是指，使用 trainer 进行训练的过程。
        :param rank: 多 GPU 训练时, 当前进程 id; 单 GPU 或 CPU 该参数为 None.
        :return: None
        """
        TorchDist.init_process_group(backend=self._distributed_parameter.backend,
                                     world_size=len(self._devices),
                                     rank=rank,
                                     init_method=self._distributed_parameter.url)

        self._start(rank=rank, device=self._devices[rank])

    def _start(self, rank: Optional[int], device: torch.device) -> None:
        """
        启动训练过程的函数,子类中实现完整的训练过程。一般是指，使用 trainer 进行训练的过程。
        :param rank: 多 GPU 训练时, 当前进程 id; 单 GPU 或 CPU 该参数为 None.
        :param device: 当前使用的 device
        :return: None
        """
        raise NotImplementedError()

    def _preprocess(self):
        """
        预处理，当多 GPU 训练时候，在主进程中进行的，所以，在训练前一些路径处理等，可以在此处理
        :return:
        """
        pass

    def __call__(self):

        devices = self._devices
        if len(devices) > 1:
            devices_str = ",".join([str(device) for device in devices])
            logging.info(f"开始在 {devices_str} 上训练...")
            torch.multiprocessing.spawn(fn=self._start_process,
                                        nprocs=len(devices))
        elif len(devices) == 1:
            logging.info(f"开始在 {devices[0]} 上训练...")
            self._start(rank=None, device=devices[0])
        else:
            logging.info(f"开始在 cpu 上训练...")
            self._start(rank=None, device=torch.device(type="cpu"))
