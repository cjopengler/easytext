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

from easytext.component.register import Register


class ComponentFactory:
    """
    Component Factory
    """

    def create(self, obj_config: OrderedDict, *args, **kwargs):

        assert isinstance(obj_config, OrderedDict), f"param_dict type: {type(param_dict)} 不是 OrderedDict"

        register = Register()

        for obj_name, param_dict in obj_config.items():

            param_dict: OrderedDict = param_dict
            cls_type = param_dict.pop("type")

            name_space = param_dict.pop("name_space")

            if name_space is None:
                name_space = obj_name

            if cls_type == "__object__":
                param_dict[obj_name] = register.find_object(name_space=name_space, obj_name=obj_name)
            else:
                cls = register.find_class(name=cls_type, name_space=name_space)

                assert cls is not None, f"{name_space}:{cls_type} 没有被注册"

                for k, v in param_dict.items():

                    if isinstance(v, OrderedDict):
                        v_obj = self.create(param_dict=v)
                        param_dict[k] = v_obj

                obj_config[obj_name] = cls(**param_dict)

        return obj_config


