#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
模型输出

Authors: panxu(panxu@baidu.com)
Date:    2020/05/18 10:41:00
"""

import torch


class ModelOutputs:
    """
    模型的前向运算输出结果
    """

    def __init__(self, logits: torch.Tensor):
        """
        模型输出
        :param logits: 指模型的输出用来计算 loss
        """
        self.logits = logits
