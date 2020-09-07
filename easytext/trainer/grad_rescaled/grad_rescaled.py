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


class GradRescaled:
    """
    对梯度进行调整的接口
    """

    def __call__(self, model: Module):
        """
        对 model 中的参数进行 rescaled 操作
        :param model:
        :return:
        """
        raise NotImplemented()
