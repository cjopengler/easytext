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
from easytext.model import ModelInputs


class ModelCollate:
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
