#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
mrc lr scheduler

Authors: PanXu
Date:    2021/11/05 09:17:00
"""
import torch
from torch.optim import Optimizer

from easytext.model import Model
from easytext.optimizer import LRSchedulerFactory


class MRCLrScheduler(LRSchedulerFactory):

    def __init__(self, max_lr: float, final_div_factor: float, total_steps: int = None, pct_start: float = 0):
        self.max_lr = max_lr
        self.pct_start = pct_start
        self.final_div_factor = final_div_factor
        self.total_steps = total_steps

    def create(self, optimizer: Optimizer, model: Model) -> "lr_scheduler":
        """
        创建 scheduler
        :param optimizer: 优化器
        :param model: 模型
        :return: lr scheduler
        """

        # lr scheduler, OneCycleLR 可以帮助跳出鞍点  https://zhuanlan.zhihu.com/p/350712244
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            pct_start=self.pct_start,
            final_div_factor=self.final_div_factor,
            total_steps=self.total_steps,
            anneal_strategy='linear'
        )

        return scheduler
