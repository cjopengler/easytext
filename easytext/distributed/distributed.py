#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
分布式接口

Authors: PanXu
Date:    2020/11/29 17:52:00
"""


class Distributed:
    """
    如果组件要兼容分布式和非分布式两种计算, 需要继承这个类
    这个类提供了 is_distributed 属性，在需要进行分布式计算的地点进行处理
    """

    def __init__(self, is_distributed: bool):
        """
        初始化
        :param is_distributed: True: 分布式运算标记; False: 非分布式运算标记
        :return:
        """
        self._is_distributed = is_distributed

    @property
    def is_distributed(self) -> bool:
        """
        当前是否是分布式的状态属性
        :return: True: 当前是分布式; False: 当前不是分布式
        """
        return self._is_distributed
