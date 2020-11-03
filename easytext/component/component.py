#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
组件

Authors: PanXu
Date:    2020/11/01 12:48:00
"""


class Component:
    """
    组件, 在配置文件中出现的对象可能会需要继承该类，该类在创建对象的时候是区分是否是 training 状态
    """

    def __init__(self, is_training: bool):
        """
        创建对象的初始化函数
        :param is_training: True: 表示训练状态; False: 表示非训练状态. 该参数通过 self._is_training 能够获取到
        """
        self._is_training = is_training


