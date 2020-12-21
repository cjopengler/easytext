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
    在进行分布式或者多 GPU 会校验是否该接口子类
    """

    @property
    def is_distributed(self) -> bool:
        """
        当前是否是分布式/多GPU 运算
        :return: True: 是多GPU; False: 单 GPU 或 CPU
        """
        raise NotImplementedError()
