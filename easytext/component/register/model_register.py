#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
模型组件注册器

Authors: PanXu
Date:    2020/10/27 15:03:00
"""

from easytext.component.register import ComponentRegister


class ModelRegister(ComponentRegister):
    """
    model 组件注册器
    """

    @classmethod
    def name_space(cls) -> str:
        return "model"


