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

from typing import List, Union, Optional, Dict
import logging

import torch
from torch import distributed as TorchDist

from easytext.trainer import Config
from easytext.distributed import ProcessGroupParameter


class Launcher:
    """
    训练启动器
    """

    def __init__(self):
        """
        启动器初始化
        """
        self._devices: Union[List[int], List[str]] = self._init_devices()

    def _init_devices(self) -> Union[List[str], List[int]]:
        """
        返回训练需要用到的设备
        :return: device, 如果是单一设备，返回的也是 List.
            如果是 List[str], ["cuda:0", "cuda:1"]; 如果是 List[int], [0, 1]
        """
        raise NotImplementedError()

    def _init_process_group_parameter(self,
                                      rank: Optional[int]) -> Optional[ProcessGroupParameter]:
        """
        初始化 多GPU/分布式参数
        :param rank: 多 GPU 训练时, 当前进程 id; 单 GPU 或 CPU 该参数为 None.
        :return: 进程组参数
        """
        raise NotImplementedError

    def _start_process(self, rank: Optional[int], world_size: int, devices: Union[List[str], List[int]]) -> None:
        """
        启动训练过程的函数,子类中实现完整的训练过程。一般是指，使用 trainer 进行训练的过程。
        :param rank: 多 GPU 训练时, 当前进程 id; 单 GPU 或 CPU 该参数为 None.
        :param world_size: 全部的训练进程数量, 如果是单 GPU 或 CPU 训练, 那么该值为 1
        :param devices: 训练用的 devices
        :return: None
        """

        process_group_parameter = self._init_process_group_parameter(rank=rank)

        TorchDist.init_process_group(backend=process_group_parameter.backend,
                                     world_size=world_size,
                                     rank=rank,
                                     init_method=process_group_parameter.url)

        device = torch.device(devices[rank])
        self._start(rank=rank, world_size=world_size, device=device)

    def _start(self, rank: Optional[int], world_size: int, device: torch.device) -> None:
        """
        启动训练过程的函数,子类中实现完整的训练过程。一般是指，使用 trainer 进行训练的过程。
        :param rank: 多 GPU 训练时, 当前进程 id; 单 GPU 或 CPU 该参数为 None.
        :param world_size: 全部的训练进程数量, 如果是单 GPU 或 CPU 训练, 那么该值为 1
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

        self._preprocess()

        devices = self._devices

        if len(devices) > 1:
            devices_str = ", ".join([str(device) for device in devices])
            logging.info(f"开始在 {devices_str} 上训练...")
            torch.multiprocessing.spawn(fn=self._start_process,
                                        args=(len(devices), devices),
                                        nprocs=len(devices))
        elif len(devices) == 1:
            logging.info(f"开始在 {devices[0]} device 上训练...")
            self._start(rank=None, world_size=1, device=torch.device(devices[0]))
        else:
            logging.info(f"开始在 cpu 上训练...")
            self._start(rank=None, world_size=1, device=torch.device(type="cpu"))

