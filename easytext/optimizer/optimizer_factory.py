#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
优化器创建工厂

Authors: panxu(panxu@baidu.com)
Date:    2020/06/01 16:11:00
"""
from torch.optim import Optimizer
from torch.optim import lr_scheduler
from easytext.model import Model


class OptimizerFactory:
    """
    Optimizer 创建工厂
    """

    def create(self, model: Model) -> "Optimizer":
        """
        创建 optimizer
        :param model:
        :return:
        """
        raise NotImplementedError()
