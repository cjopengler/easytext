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
from torch.nn.utils import clip_grad_norm_

from .grad_rescaled import GradRescaled


class GradNormalize(GradRescaled):
    """
    对梯度 进行 normalize
    """

    def __init__(self, max_normalize: float):
        """
        初始化
        :param max_normalize: 最大的正则化值，一般会设置为 5 或者 10.
        """

        self._max_normalize = max_normalize

        assert self._max_normalize > 0., f"{self._max_normalize} 非法, 应该大于 0"

    def __call__(self, model: Module):
        clip_grad_norm_(model.parameters(), self._max_normalize)

