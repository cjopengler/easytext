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

from easytext.component import ComponentBuiltinKey
from easytext.component.register import ComponentRegister
from easytext.component.register.component_register import T


class BuiltinRegister:
    """
    easytext 系统使用的注册器，外部永远不要使用这个注册器，而应该使用 ComponentRegister
    """

    @classmethod
    def register_class(cls, name: str, is_allowed_exist: bool = False) -> T:
        return ComponentRegister.register_class(name, ComponentBuiltinKey.BUILTIN_NAME_SPACE, is_allowed_exist)


