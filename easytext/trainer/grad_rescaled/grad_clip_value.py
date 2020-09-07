#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
对参数的相关梯度进行调整

Authors: PanXu
Date:    2020/08/21 09:34:00
"""

from torch.nn import Module
from torch.nn.utils import clip_grad_value_
from easytext.model import Model

from .grad_rescaled import GradRescaled


class GradClipValue(GradRescaled):
    """
    对梯度进行 clip
    """

    def __init__(self, clip_value: float):
        """
        初始化
        :param clip_value: 梯度进行 clip 的 value, 区间是 [-clip_value, clip_value]. clip_value 值大于 0.
        一般会设置为 5 或者 10.
        """
        self._clip_value = clip_value

        assert self._clip_value > 0., f"{self._clip_value} 非法，应该大于 0"

    def __call__(self, model: Module):
        clip_grad_value_(model.parameters(), self._clip_value)

