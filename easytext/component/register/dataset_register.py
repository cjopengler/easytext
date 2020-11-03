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
from easytext.component.register.component_register import T


class DatasetRegister:
    """
    model 组件注册器
    """
    NAME_SPACE = "dataset"

    @classmethod
    def register_class(cls, name: str, is_allowed_exist: bool = False) -> T:
        return ComponentRegister.register(name, DatasetRegister.NAME_SPACE, is_allowed_exist)


