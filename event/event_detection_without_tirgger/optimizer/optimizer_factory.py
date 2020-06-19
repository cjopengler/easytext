#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
优化器

Authors: panxu(panxu@baidu.com)
Date:    2020/06/17 22:20:00
"""
from torch.optim import Optimizer
from torch.optim import SGD
from easytext.model import Model
from easytext.optimizer import OptimizerFactory


class EventOptimizerFactory(OptimizerFactory):
    """
    优化器工厂
    """

    def create(self, model: Model) -> "Optimizer":
        optimizer = SGD(params=model.parameters(),
                        lr=0.01)
        return optimizer
