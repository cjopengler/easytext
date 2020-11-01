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
from typing import Dict
from collections import OrderedDict

from torch.nn import Linear

from easytext.model import Model
from easytext.data import Vocabulary
from easytext.component import Component
from easytext.component.register import ComponentRegister, ModelRegister
from easytext.component.factory import ComponentFactory

from easytext.tests import ASSERT
from easytext.tests import ROOT_PATH


@ModelRegister.register_class(name="_MyModel")
class _MyModel(Model):

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = Linear(in_features=input_size, out_features=output_size)


@ModelRegister.register_class(name="_NestedModel")
class _NestedModel(Model):

    def __init__(self, input_size: int, output_size: int, my_model: _MyModel):
        super().__init__()
        self.sub_model = Linear(in_features=input_size, out_features=output_size)
        self.my_model = my_model

    def reset_parameters(self):
        pass


@ComponentRegister.register_class(name="_CustomerObj", name_space="test")
class _CustomerObj(Component):

    def __init__(self, is_training: bool, value: int):
        super().__init__(is_training=is_training)

        self.value = value


@ModelRegister.register_class(name="_ModelWithObjParam")
class _ModelWithObjParam(Model):

    def __init__(self, input_size: int, output_size: int, customer_obj: _CustomerObj):
        super().__init__()
        self.sub_model = Linear(in_features=input_size, out_features=output_size)
        self.customer_obj = customer_obj

    def reset_parameters(self):
        pass


@ComponentRegister.register_class(name="_DictParamComponent", name_space="test")
class _DictParamComponent(Component):

    def __init__(self, is_training: bool, dict_value: Dict, customer_obj: _CustomerObj):
        super().__init__(is_training=is_training)
        self.dict_value = dict_value
        self.curstomer_obj = customer_obj


@ComponentRegister.register_class(name="_TrainingComponent", name_space="training_component")
class _TrainingComponent(Component):
    """
    带有 trainging 的Component
    """

    def __init__(self, value: int, is_training: bool):
        super().__init__(is_training=is_training)

        if is_training:
            self.value = f"training_{value}"
        else:
            self.value = f"evaluate_{value}"


def test_component_factory():
    config_dict = OrderedDict(
        {
            "model":
                {
                    "__type__": "_MyModel",
                    "__name_space__": "model",
                    "input_size": 2,
                    "output_size": 4
                }
        })

    factory = ComponentFactory(is_training=True)

    parserd_dict = factory.create(config=config_dict)

    model = parserd_dict["model"]

    ASSERT.assertTrue(model.linear is not None)
    ASSERT.assertEqual((2, 4), (model.linear.in_features, model.linear.out_features))


def test_component_nested_factory():

    nested_json_file_path = "data/easytext/tests/component/nested.json"
    nested_json_file_path = os.path.join(ROOT_PATH, nested_json_file_path)
    with open(nested_json_file_path, encoding="utf-8") as f:

        param_dict = json.load(f, object_pairs_hook=OrderedDict)

    factory = ComponentFactory(is_training=True)

    parsed_dict = factory.create(config=param_dict)

    model = parsed_dict["model"]

    ASSERT.assertTrue(model.sub_model is not None)
    ASSERT.assertEqual((2, 4), (model.sub_model.in_features, model.sub_model.out_features))

    ASSERT.assertTrue(model.my_model is not None)
    ASSERT.assertTrue(model.my_model.linear is not None)
    ASSERT.assertEqual((3, 6), (model.my_model.linear.in_features, model.my_model.linear.out_features))


def test_component_training_factory():

    config_json_file_path = "data/easytext/tests/component/training.json"
    config_json_file_path = os.path.join(ROOT_PATH, config_json_file_path)
    with open(config_json_file_path, encoding="utf-8") as f:

        param_dict = json.load(f, object_pairs_hook=OrderedDict)

    factory = ComponentFactory(is_training=True)

    parsed_dict = factory.create(config=param_dict)

    my_component = parsed_dict["my_component"]

    ASSERT.assertEqual("training_3", my_component.value)


def test_component_evaluate_factory():

    config_json_file_path = "data/easytext/tests/component/training.json"
    config_json_file_path = os.path.join(ROOT_PATH, config_json_file_path)
    with open(config_json_file_path, encoding="utf-8") as f:

        param_dict = json.load(f, object_pairs_hook=OrderedDict)

    factory = ComponentFactory(is_training=False)

    parsed_dict = factory.create(config=param_dict)

    my_component = parsed_dict["my_component"]

    ASSERT.assertEqual("evaluate_3", my_component.value)


def test_component_with_object():
    """
    测试，当 component 构建的时候，某个参数是 object
    :return:
    """

    config_json_file_path = "data/easytext/tests/component/component_with_obj.json"
    config_json_file_path = os.path.join(ROOT_PATH, config_json_file_path)
    with open(config_json_file_path, encoding="utf-8") as f:
        param_dict = json.load(f, object_pairs_hook=OrderedDict)

    factory = ComponentFactory(is_training=False)

    parsed_dict = factory.create(config=param_dict)

    my_obj = parsed_dict["my_obj"]

    ASSERT.assertEqual(10, my_obj.value)

    my_component: _ModelWithObjParam = parsed_dict["my_component"]

    ASSERT.assertEqual(4, my_component.sub_model.in_features)
    ASSERT.assertEqual(2, my_component.sub_model.out_features)

    ASSERT.assertTrue(id(my_obj) == id(my_component.customer_obj))

    another_component: _ModelWithObjParam = parsed_dict["another_component"]

    ASSERT.assertTrue(id(my_component) != id(another_component))

    another_obj: _CustomerObj = parsed_dict["another_obj"]
    ASSERT.assertTrue(id(another_obj) == id(another_component.customer_obj))

    ASSERT.assertEqual(20, another_obj.value)

    dict_param_component: _DictParamComponent = parsed_dict["dict_param_component"]
    ASSERT.assertTrue(id(dict_param_component.curstomer_obj) == id(another_obj))

    ASSERT.assertEqual(1, dict_param_component.dict_value["a"])
    ASSERT.assertEqual(2, dict_param_component.dict_value["b"])
    ASSERT.assertEqual(30, dict_param_component.dict_value["c_obj"].value)

