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

from typing import TypeVar, Type

from easytext.component.register import ClassRegister, ObjectRegister, Register
from easytext.component.register.class_register import T


class ComponentRegister:
    """
    Component 注册器，也是装饰器
    """

    @classmethod
    def register_class(cls, name: str, is_allowed_exist: bool = False) -> None:
        """
        用作在类上的装饰器
        :param name: 在配置文件中的 名字
        :param is_allowed_exist: True: 允许名字重复，那么，后面的名字会将前面的名字覆盖, 正常来讲不应该出现这样的设置;
                                 False: 不允许名字重复, 如果出现重复，自己定义的名字需要进行修改
        :return:
        """
        register = Register()

        def add_class_to_registry(registed_class: Type[T]):
            register.register_class(cls=registed_class,
                                    name_space=cls.name_space(),
                                    name=name,
                                    is_allowed_exist=is_allowed_exist)

        return add_class_to_registry

    @classmethod
    def register_object(cls, name: str, is_allowed_exist: bool = False) -> None:
        """
        用作在函数上的装饰器, 会将函数的返回值作为 obj 放入到 object 注册表中
        :param name: 在配置文件中的 名字
        :param is_allowed_exist: True: 允许名字重复，那么，后面的名字会将前面的名字覆盖, 正常来讲不应该出现这样的设置;
                                 False: 不允许名字重复, 如果出现重复，自己定义的名字需要进行修改
        :return:
        """
        register = Register()

        def add_object_to_registry(obj: T):
            register.register_object(obj=obj,
                                     name_space=cls.name_space(),
                                     name=name,
                                     is_allowed_exist=is_allowed_exist)

        return add_object_to_registry

    @classmethod
    def name_space(cls) -> str:
        raise NotImplementedError()
