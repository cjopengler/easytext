#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 component factory

Authors: PanXu
Date:    2020/10/27 17:46:00
"""
import json
import os
from collections import OrderedDict
from torch.nn import Linear

from easytext.model import Model
from easytext.data import Vocabulary
from easytext.component.register import ModelRegister
from easytext.component.factory import ComponentFactory

from easytext.tests import ASSERT
from easytext.tests import ROOT_PATH


@ModelRegister.register_class(name="my_model")
class _MyModel(Model):

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = Linear(in_features=input_size, out_features=output_size)


@ModelRegister.register_class(name="nested_model")
class _NestedModel(Model):

    def __init__(self, input_size: int, output_size: int, my_model: _MyModel):
        super().__init__()
        self.sub_model = Linear(in_features=input_size, out_features=output_size)
        self.my_model = my_model


@ModelRegister.register_class(name="model_with_obj_param")
class _ModelWithObjParam(Model):

    def __init__(self, input_size: int, output_size: int, vocabulary: Vocabulary = None):
        super().__init__()
        self.sub_model = Linear(in_features=input_size, out_features=output_size)
        self.vocabulary = vocabulary


def test_component_factory():
    config_dict = OrderedDict(
        {
            "model":
                {
                    "__type__": "my_model",
                    "__name_space__": "model",
                    "input_size": 2,
                    "output_size": 4
                }
        })

    factory = ComponentFactory()

    parserd_dict = factory.create(config=config_dict)

    model = parserd_dict["model"]

    ASSERT.assertTrue(model.linear is not None)
    ASSERT.assertEqual((2, 4), (model.linear.in_features, model.linear.out_features))


def test_component_nested_factory():

    nested_json_file_path = "data/easytext/tests/component/nested.json"
    nested_json_file_path = os.path.join(ROOT_PATH, nested_json_file_path)
    with open(nested_json_file_path, encoding="utf-8") as f:

        param_dict = json.load(f, object_pairs_hook=OrderedDict)

    factory = ComponentFactory()

    parsed_dict = factory.create(config=param_dict)

    model = parsed_dict["model"]

    ASSERT.assertTrue(model.sub_model is not None)
    ASSERT.assertEqual((2, 4), (model.sub_model.in_features, model.sub_model.out_features))

    ASSERT.assertTrue(model.my_model is not None)
    ASSERT.assertTrue(model.my_model.linear is not None)
    ASSERT.assertEqual((3, 6), (model.my_model.linear.in_features, model.my_model.linear.out_features))

