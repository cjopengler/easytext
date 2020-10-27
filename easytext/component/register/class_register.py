#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
对类进行注册

Authors: PanXu
Date:    2020/10/27 14:54:00
"""
from typing import TypeVar, Type


T = TypeVar('T')


class ClassRegister:
    """
    类的注册器
    """

    def register_class(self, cls: Type[T], name_space: str, name: str, is_allowed_exist: bool = False) -> None:
        """
        对类进行注册
        :param cls: 类
        :param name_space: name sapce 范围, 比如 model, optimizer 为了减轻重名问题
        :param name: 定义的名, 会在配置文件中用到的名字
        :param is_allowed_exist: True: 允许名字重复，那么，后面的名字会将前面的名字覆盖, 正常来讲不应该出现这样的设置;
                                 False: 不允许名字重复, 如果出现重复，自己定义的名字需要进行修改
        :return:
        """
        raise NotImplementedError()

    def find_class(self, name_space: str, name: str) -> Type[T]:
        raise NotImplementedError
