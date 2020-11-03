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

    def __init__(self, is_training:bool, config_file_path: str, encoding="utf-8"):
        with open(config_file_path, encoding=encoding) as f:
            config_dict = json.load(f, object_pairs_hook=OrderedDict)

            component_factory = ComponentFactory(is_training=is_training)

            self._config = component_factory.create(config_dict)

    def __getattr__(self, item):
        if item in self._config:
            return self._config[item]
        raise AttributeError(f"{item} 属性不存在!")

