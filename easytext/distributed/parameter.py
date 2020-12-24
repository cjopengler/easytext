#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
分布式使用的参数

Authors: PanXu
Date:    2020/12/08 20:34:00
"""

from easytext.component.register import EasytextRegister


@EasytextRegister.register()
class ProcessGroupParameter:
    """
    分布式训练初始化进程组使用的参数
    """

    def __init__(self, port: int, backend: str = "nccl", host: str = "127.0.0.1"):
        """
        分布式训练时候使用的相关参数
        :param port: 本机空闲的端口, 用作 init_process_group 中 init_method 参数使用
        :param backend: distributed 使用的 backend, "nccl" 或者 "gloo"
        :param host: 主机地址
        """
        self.backend = backend
        self.port = port
        self.host = host
        self.url = f"tcp://{self.host}:{self.port}"


@EasytextRegister.register()
class DistributedDataParallelParameter:
    """
    分布式训练使用的参数
    """

    def __init__(self, find_unused_parameters: bool):
        """
        DistributedDataParallel, 初始化用到的参数, 这里放置的是在训练中需要用户配置的。
        该类中没有的，会自动进行设置
        """
        self.find_unused_parameters = find_unused_parameters
