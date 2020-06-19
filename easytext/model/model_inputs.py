#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
模型输入

Authors: panxu(panxu@baidu.com)
Date:    2020/06/10 09:55:00
"""
from typing import Dict

import torch


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
