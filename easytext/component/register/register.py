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

from typing import Dict, Type

from easytext.component.register import ClassRegister, ObjectRegister
from easytext.component.register.class_register import T


class Register(ClassRegister, ObjectRegister):
    """
    全局注册器，单件模式
    """

    __instance = None

    def __new__(cls, *args, **kwargs):
        if not cls.__instance:
            cls.__instance = super().__new__(cls)

            # 类注册表
            cls._class_registry: Dict[str, Dict[str, Type[T]]] = dict()
            # object 注册表
            cls._object_registry: Dict[str, Dict[str, T]] = dict()

        return cls.__instance

    def __init__(self):
        # 这里不应该进行任何初始化行为
        pass

    def register_class(self, cls: Type[T], name_space: str, name: str, is_allowed_exist: bool = False) -> None:

        name_space_dict = self._class_registry.get(name_space, dict())

        if name in name_space_dict and not is_allowed_exist:
            raise RuntimeError("")

        name_space_dict[name] = cls

        self._class_registry[name_space] = name_space_dict

    def register_object(self, obj: T, name_space: str, name: str, is_allowed_exist: bool = False) -> None:
        name_space_dict = self._object_registry.get(name_space, dict())

        if name in name_space_dict and not is_allowed_exist:
            raise RuntimeError("")

        name_space_dict[name] = obj

        self._object_registry[name_space] = name_space_dict

    def find_class(self, name_space: str, name: str) -> Type[T]:
        return self._class_registry.get(name_space, dict()).get(name, None)

    def find_object(self, name_space: str, name: str) -> T:
        return self._object_registry.get(name_space, dict()).get(name, None)

