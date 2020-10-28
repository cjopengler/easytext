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
from collections import OrderedDict
from torch.nn import Linear

from easytext.model import Model
from easytext.data import Vocabulary
from easytext.component.register import ModelRegister
from easytext.component.factory import ComponentFactory


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

    param_dict = OrderedDict({"model": {
        "__type__": "my_model",
        "__name_space__": "model",
        "input_size": 2,
        "output_size": 4
    }})

    factory = ComponentFactory()

    config_obj = factory.create(obj_config=param_dict)

    print(config_obj)


def test_component_nested_factory():

    param_dict = OrderedDict({"model": {
        "__type__": "nested_model",
        "__name_space__": "model",
        "input_size": 2,
        "output_size": 4,

        "my_model": {
            "__type__": "my_model",
            "__name_space__": "model",
            "input_size": 3,
            "output_size": 6
        }
    }})

    factory = ComponentFactory()

    config_obj = factory.create(obj_config=param_dict)

    print(config_obj)


def test_component_object_factory():

    @ModelRegister.register_object("vocabulary")
    def create_vocabulary():
        return Vocabulary(tokens=[["A", "B", "C"]])

    param_dict = OrderedDict({"model": {
        "__type__": "model_with_obj_param",
        "__name_space__": "model",
        "input_size": 2,
        "output_size": 4,

        "vocabulary": {
            "__type__": "__object__",
            "name_space": "model"
        }
    }})

    factory = ComponentFactory()

    config_obj = factory.create(obj_config=param_dict)

    print(config_obj)

