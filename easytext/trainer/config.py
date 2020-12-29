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
from typing import Dict
from collections import OrderedDict
import json
import copy

from easytext.component.factory import ComponentFactory


class Config:
    """
    训练器的配置
    """

    def __init__(self, is_training: bool, config_file_path: str, encoding: str = "utf-8"):

        with open(config_file_path, encoding=encoding) as f:
            self._raw_config: OrderedDict = json.load(f, object_pairs_hook=OrderedDict)
            self._is_training = is_training
            component_factory = ComponentFactory(self._is_training)
            self._config = component_factory.create(self._raw_config)

    def __str__(self):
        return json.dumps(self._raw_config, ensure_ascii=False)

    def __getattr__(self, key):
        return self._config[key]

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_config"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        component_factory = ComponentFactory(self._is_training)
        self._config = component_factory.create(self._raw_config.copy())

