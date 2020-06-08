#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
learning rate scheduler factory

Authors: panxu(panxu@baidu.com)
Date:    2020/06/03 09:58:00
"""

from torch.optim import Optimizer
from torch.optim import lr_scheduler
from easytext.model import Model


class LRSchedulerFactory:

    def create(self, optimizer: Optimizer, model: Model) -> "lr_scheduler":
        """
        创建 Learning Rate Scheduler
        :param optimizer: 优化器
        :param model: 模型
        :return: lr scheduler 对象
        """
        raise NotImplementedError()
