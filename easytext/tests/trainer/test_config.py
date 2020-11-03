#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
对 trainer config 进行测试

Authors: PanXu
Date:    2020/10/29 10:18:00
"""
import os

from easytext.component.register import ComponentRegister
from easytext.trainer import Config

from easytext.tests import ROOT_PATH
from easytext.tests import ASSERT


@ComponentRegister.register(typename="Optimizer", name_space="optimizer")
class _MyOpitmizer:
    pass


@ComponentRegister.register(typename="MyModel", name_space="model")
class _MyModel:

    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size


def test_config():

    config_file_path = "data/easytext/tests/config/config.json"
    config_file_path = os.path.join(ROOT_PATH, config_file_path)

    config = Config(is_training=True,
                    config_file_path=config_file_path)

    ASSERT.assertTrue(config.model is not None)
    ASSERT.assertTrue(isinstance(config.model, _MyModel))

    ASSERT.assertTrue(config.optimizer is not None)
    ASSERT.assertTrue(isinstance(config.optimizer, _MyOpitmizer))
