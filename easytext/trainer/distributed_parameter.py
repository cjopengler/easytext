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

    def __init__(self, backend: str, free_port: int):
        """
        分布式训练时候使用的相关参数
        :param backend: distributed 使用的 backend, 如果 devices 不是多 gpu, 该参数是 None; 否则, 应该是 "nccl" 或者 "gloo"
        :param free_port: 本机空闲的端口, 用作 init_process_group 中 init_method 参数使用
        """
        self.backend = backend
        self.free_port = free_port
