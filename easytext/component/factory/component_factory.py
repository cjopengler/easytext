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

from typing import OrderedDict

from easytext.component.component_builtin_key import ComponentBuiltinKey
from easytext.component.register import Register


class ComponentFactory:
    """
    Component Factory
    """

    def create(self, obj_config: OrderedDict, *args, **kwargs):

        assert isinstance(obj_config, OrderedDict), f"param_dict type: {type(obj_config)} 不是 OrderedDict"

        register = Register()

        for obj_name, param_dict in obj_config.items():

            param_dict: OrderedDict = param_dict
            component_type = param_dict.pop(ComponentBuiltinKey.TYPE)

            name_space = param_dict.pop(ComponentBuiltinKey.NAME_SPACE)

            if component_type == ComponentBuiltinKey.OBJECT_TYPE:
                param_dict[obj_name] = register.find_object(name_space=name_space, obj_name=obj_name)
            else:
                cls = register.find_class(name=component_type, name_space=name_space)

                assert cls is not None, f"{name_space}:{component_type} 没有被注册"

                for k, v in param_dict.items():

                    if isinstance(v, OrderedDict):
                        v_obj = self.create(param_dict=v)
                        param_dict[k] = v_obj

                obj_config[obj_name] = cls(**param_dict)

        return obj_config


