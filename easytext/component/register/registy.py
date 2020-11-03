#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
注册器

Authors: PanXu
Date:    2020/10/27 15:11:00
"""

from typing import Dict, Type, TypeVar

T = TypeVar('T')


class Registry:
    """
    全局注册器，单件模式
    """

    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)

            # 类注册表
            cls._class_registry: Dict[str, Dict[str, Type[T]]] = dict()

            # 对象注册表
            cls._object_registry: Dict[str, Dict[str, T]] = dict()

        return cls.__instance

    def __init__(self):
        # 这里不应该进行任何初始化行为
        pass

    def register_class(self, cls: Type[T], name_space: str, name: str, is_allowed_exist: bool = False) -> "Registry":
        """
        注册类
        :param cls: 要注册的 类
        :param name_space: 命名空间
        :param name: 配置文件中的名字
        :param is_allowed_exist: True: 如果名字和命名空间一样, 会覆盖存在的类; False: 会报错
        :return:
        """

        name_space_dict = self._class_registry.get(name_space, dict())

        if name in name_space_dict and not is_allowed_exist:
            raise RuntimeError(f"{name_space}.{name} {cls} 已经存在!")

        name_space_dict[name] = cls

        self._class_registry[name_space] = name_space_dict
        return self

    def find_class(self, name_space: str, name: str) -> Type[T]:
        """
        根据 name space 或 name 找到 class 或 function
        :param name_space: 类在config文件中的命名空间
        :param name: 泪在 config 文件中的 类型
        :return:
        """
        return self._class_registry.get(name_space, dict()).get(name, None)

    def register_object(self, obj: T, name: str) -> "Registry":
        """
        注册 object
        :param obj: 实例对象
        :param name: 实例对象的名字
        :return:
        """

        if name in self._object_registry:
            raise RuntimeError(f"{name} object 已经存在!")

        self._object_registry[name] = obj

        return self

    def find_object(self, name: str) -> T:
        """
        根据配置文件中的名字找到对象
        :param name: 对象的名字
        :return:
        """
        return self._object_registry[name]

    def clear_objects(self) -> "Registry":
        """
        清空注册表
        :return:
        """
        self._object_registry.clear()

        return self


