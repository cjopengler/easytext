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
class DistributedParameter:
    """
    分布式训练使用的参数
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
