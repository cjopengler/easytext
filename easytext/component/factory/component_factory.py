#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
Component Factory

Authors: PanXu
Date:    2020/10/27 16:14:00
"""
from collections import OrderedDict

from easytext.component.component_builtin_key import ComponentBuiltinKey
from easytext.component.register import Registry


class ComponentFactory:
    """
    Component Factory
    """

    def __init__(self):
        self._registry = Registry()

    def _create(self, param_dict: OrderedDict):

        param_dict: OrderedDict = param_dict

        if ComponentBuiltinKey.TYPE not in param_dict:
            raise RuntimeError(f"{ComponentBuiltinKey.TYPE} 没有在 json 中设置")

        if ComponentBuiltinKey.NAME_SPACE not in param_dict:
            raise RuntimeError(f"{ComponentBuiltinKey.NAME_SPACE} 没有在 json 中设置")

        component_type = param_dict.pop(ComponentBuiltinKey.TYPE)
        name_space = param_dict.pop(ComponentBuiltinKey.NAME_SPACE)

        cls = self._registry.find_class(name=component_type, name_space=name_space)

        if cls is None:
            raise RuntimeError(f"{name_space}:{component_type} 没有被注册")

        for param_name, param_value in param_dict.items():

            if isinstance(param_value, OrderedDict):
                v_obj = self._create(param_dict=param_value)
                param_dict[param_name] = v_obj
            else:
                # 不用处理
                pass

        return cls(**param_dict)

    def create(self, config: OrderedDict):
        """
        创建对象工厂
        :param config: config 字典, 是 OrderedDict, 其中的 key 会按照顺序执行
        :return:
        """

        assert isinstance(config, OrderedDict), f"param_dict type: {type(config)} 不是 OrderedDict"

        parsed_config = OrderedDict(config)

        for obj_name, param_dict in parsed_config.items():
            parsed_config[obj_name] = self._create(param_dict=param_dict)
        return parsed_config
