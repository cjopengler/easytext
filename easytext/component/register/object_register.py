#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
brief

Authors: PanXu
Date:    2020/10/27 14:54:00
"""

from typing import TypeVar


T = TypeVar('T')


class ObjectRegister:
    """
    对象注册器
    """

    def register_object(self, obj: T, name_space: str, name: str, is_allowed_exist: bool = False) -> None:
        """
        注册对象
        :param obj: 注册的对象
        :param name_space: 对象所属的命名空间，避免重名
        :param name: 对象名字, 在配置文件中写名字
        :param is_allowed_exist: True: 允许名字重复，那么，后面的名字会将前面的名字覆盖, 正常来讲不应该出现这样的设置;
                                 False: 不允许名字重复, 如果出现重复，自己定义的名字需要进行修改
        :return:
        """
        raise NotImplementedError()

    def find_object(self, name_space: str, name: str) -> T:
        raise NotImplementedError()
