#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
数据集基类

Authors: PanXu
Date:    2021/10/18 17:19:00
"""
from typing import Dict, Union, List

from torch.utils.data import Dataset as TorchDataset

from easytext.data import InstanceFactory, Instance
from easytext.component import Component


class Dataset(TorchDataset, InstanceFactory, Component):
    """
    easytext 数据集，继承了
    1. torch.utils.data.Dataset
    2. easytext.component.Component 用作训练阶段和预测阶段的不同处理
    3. easytext.data.InstanceFactory 保证以同样的方式来创建 instance
    """

    def __init__(self, is_training: bool):
        TorchDataset.__init__(self)
        Component.__init__(self, is_training=is_training)

    def create_instance(self, input_data: Dict) -> Union[Instance, List[Instance]]:
        raise NotImplementedError()
