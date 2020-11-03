#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
组件注册器

Authors: PanXu
Date:    2020/10/27 15:02:00
"""

from typing import Type

from easytext.component.register import Registry
from easytext.component.register.registy import T


class ComponentRegister:
    """
    Component 注册器，也是类装饰器
    """

    @classmethod
    def register(cls, name_space: str, typename: str = None, is_allowed_exist: bool = False) -> T:
        """
        用作在类或函数上的装饰器
        :param typename: 注册的类或者函数的类型名称, 如果为 None 或者 "", 那么, 将会默认使用类或者函数的名字作为名字,
                         在配置文件中直接使用类名字或者函数名字即可。
        :param name_space: 注册的类或者函数的 name space
        :param is_allowed_exist: True: 允许名字重复，那么，后面的名字会将前面的名字覆盖, 正常来讲不应该出现这样的设置;
                                 False: 不允许名字重复, 如果出现重复，自己定义的名字需要进行修改
        :return:
        """
        register = Registry()

        def add_to_registry(registered_class: Type[T]):

            name = typename or registered_class.__name__

            register.register_class(cls=registered_class,
                                    name_space=name_space,
                                    name=name,
                                    is_allowed_exist=is_allowed_exist)
            return registered_class

        return add_to_registry
