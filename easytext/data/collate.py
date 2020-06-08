#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
数据预处理 collate_fn

Authors: panxu(panxu@baidu.com)
Date:    2020/05/25 16:13:00
"""

from typing import Iterable, Dict

import torch

from .instance import Instance


class ModelInputs:
    """
    批量的模型输入
    """

    def __init__(self,
                 batch_size: int,
                 model_inputs: Dict[str, torch.Tensor],
                 labels: torch.Tensor = None):
        """
        初始化
        :param batch_size: batch_size
        :param model_inputs: 模型输入，这个参数会被以 **model_inputs 传给模型，所以该字典的key 要与模型参数保持一致,
        相当于 X.
        :param labels: labels 如果是测试集，那么 labels=None. 相当于 Y.
        """
        self.batch_size = batch_size
        self.model_inputs = model_inputs
        self.labels = labels


class Collate:
    """
    用在对数据处理产出模型的输入数据以及label, 是 torch.utils.data.DataLoader 中的 collate_fn 函数
    """

    def __call__(self, instances: Iterable[Instance]) -> ModelInputs:
        """
        collate_fn 执行
        :param instances: torch.utils.data.Dataset 中 __getitem__ 返回的是 Instance
        :return: 模型需要调用的数据
        """
        raise NotImplementedError()
