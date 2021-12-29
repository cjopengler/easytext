#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
模型输入

Authors: PanXu
Date:    2021/10/25 18:38:00
"""
from typing import Dict

import torch

from easytext.data import ModelInputs


class MRCModelInputs(ModelInputs):
    """
    MRC model inputs
    """

    def __init__(self,
                 batch_size: int,
                 model_inputs: Dict[str, torch.Tensor],
                 labels: Dict[str, torch.Tensor] = None):
        """
        初始化
        :param batch_size: batch_size
        :param model_inputs: 模型输入
        :param labels: 由于我们有 start end 以及 match 所以是字典
        """
        super().__init__(batch_size, model_inputs, labels)




