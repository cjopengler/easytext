#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
训练的配置

Authors: PanXu
Date:    2020/10/29 10:01:00
"""
from collections import OrderedDict
import json

from easytext.component.factory import ComponentFactory


class Config:
    """
    训练器的配置
    """

    def __init__(self, is_training: bool, config_file_path: str, encoding: str = "utf-8"):
        with open(config_file_path, encoding=encoding) as f:
            config_dict = json.load(f, object_pairs_hook=OrderedDict)
            self._is_training = is_training
            component_factory = ComponentFactory(self._is_training)

            self.config_dict = component_factory.create(config_dict)

    def get(self, k):
        return self.config_dict.get(k, None)



class ConfigBak:
    """
    训练器的配置
    """

    def __init__(self, is_training: bool, config_file_path: str, encoding: str = "utf-8"):
        with open(config_file_path, encoding=encoding) as f:
            config_dict = json.load(f, object_pairs_hook=OrderedDict)
            self.__dict__.update(config_dict)
            self._is_training = is_training

    def __str__(self):
        return json.dumps(self.__config_dict, ensure_ascii=False)

    def __getattr__(self, item):
        return self.__dict__.get(item, None)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def build(self) -> "Config":
        component_factory = ComponentFactory(self._is_training)

        self.__dict__.update(component_factory.create(self.__config_dict))
        return self
