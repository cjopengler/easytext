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
import traceback
import logging
import inspect
import copy
from collections import OrderedDict

from easytext.component import Component
from easytext.component.component_builtin_key import ComponentBuiltinKey
from easytext.component.register import Registry
from easytext.utils.json_util import json2str


class ComponentFactory:
    """
    Component Factory
    """

    def __init__(self, is_training: bool):
        """
        初始化
        :param is_training: True: training 状态创建 component; False: 非training状态创建 component
        """
        self._is_training = is_training
        self._registry = Registry()

    def _create_by_object(self, path: str, param_dict: OrderedDict):
        """
        通过 object 来得到 object, 因为 object 是之前创建好的，直接过去就好
        :param path:
        :param param_dict:
        :return:
        """
        object_path = param_dict[ComponentBuiltinKey.OBJECT]
        return self._registry.find_object(object_path)

    def _create_by_type(self, path: str, param_dict: OrderedDict):
        """
        通过 type 和 name space 创建 object
        :param path:
        :param param_dict:
        :return:
        """
        component_type = param_dict.pop(ComponentBuiltinKey.TYPENAME)
        name_space = param_dict.pop(ComponentBuiltinKey.NAME_SPACE)

        cls = self._registry.find_class(name=component_type, name_space=name_space)

        if cls is None:
            raise RuntimeError(f"{name_space}:{component_type} 没有被注册")

        for param_name, param_value in param_dict.items():

            if isinstance(param_value, OrderedDict):
                sub_path = f"{path}.{param_name}"
                v_obj = self._create(path=sub_path, param_dict=param_value)
                param_dict[param_name] = v_obj
            else:
                # 不用处理
                pass

        # 增加 is_training 参数
        need_is_training_parameter = False
        if inspect.isclass(cls):
            if issubclass(cls, Component):
                need_is_training_parameter = True
            else:
                # 非 Component 类，不做任何处理
                pass
        elif inspect.isfunction(cls):
            if ComponentBuiltinKey.IS_TRAINING in inspect.getfullargspec(cls).args:
                need_is_training_parameter = True

        else:
            raise RuntimeError(f"{cls} 错误! 应该是 函数 或者是 类的静态函数, 不能是类函数或者成员函数")

        if need_is_training_parameter and (ComponentBuiltinKey.IS_TRAINING not in param_dict):
            param_dict[ComponentBuiltinKey.IS_TRAINING] = self._is_training

        try:
            obj = cls(**param_dict)
        except TypeError as type_error:
            logging.fatal(f"Exception: {type_error} for {cls}")
            logging.fatal(traceback.format_exc())
            raise type_error

        self._registry.register_object(name=path, obj=obj)
        return obj

    def _create_by_raw_dict(self, path: str, param_dict: OrderedDict):
        """
        没有 type 和 name space, 该字典就是参数
        :param path:
        :param param_dict:
        :return:
        """

        for param_name, param_value in param_dict.items():

            if isinstance(param_value, OrderedDict):
                sub_path = f"{path}.{param_name}"
                v_obj = self._create(path=sub_path, param_dict=param_value)
                param_dict[param_name] = v_obj
            else:
                # 不用处理
                pass

        return param_dict

    def _create(self, path: str, param_dict: OrderedDict):

        param_dict: OrderedDict = param_dict

        if ComponentBuiltinKey.OBJECT in param_dict:
            return self._create_by_object(path=path, param_dict=param_dict)

        elif ComponentBuiltinKey.TYPENAME in param_dict and ComponentBuiltinKey.NAME_SPACE in param_dict:
            return self._create_by_type(path=path, param_dict=param_dict)

        elif ComponentBuiltinKey.TYPENAME in param_dict and ComponentBuiltinKey.NAME_SPACE not in param_dict:
            raise RuntimeError(f"构建 {path} 错误, "
                               f"{ComponentBuiltinKey.TYPENAME} 与 {ComponentBuiltinKey.NAME_SPACE} 必须同时出现")

        elif ComponentBuiltinKey.TYPENAME not in param_dict and ComponentBuiltinKey.NAME_SPACE in param_dict:
            raise RuntimeError(f"构建 {path} 错误, "
                               f"{ComponentBuiltinKey.TYPENAME} 与 {ComponentBuiltinKey.NAME_SPACE} 必须同时出现")

        else:
            # 这种情况是指，参数就是一个字典
            return self._create_by_raw_dict(path=path, param_dict=param_dict)

    def create(self, config: OrderedDict):
        """
        创建对象工厂
        :param config: config 字典, 是 OrderedDict, 其中的 key 会按照顺序执行
        :return:
        """

        assert isinstance(config, OrderedDict), f"param_dict type: {type(config)} 不是 OrderedDict"

        parsed_config = copy.deepcopy(config)

        for obj_name, param_dict in parsed_config.items():

            if isinstance(param_dict, OrderedDict):
                parsed_config[obj_name] = self._create(obj_name, param_dict=param_dict)
            else:
                # 对于非字典的根节点下的是基础类型，不用做处理
                pass
        return parsed_config
