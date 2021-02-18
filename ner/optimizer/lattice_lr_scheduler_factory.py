#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
lattice lr scheduler factory

Authors: PanXu
Date:    2020/06/28 08:48:00
"""
from torch.optim import lr_scheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from easytext.model import Model
from easytext.optimizer.lr_scheduler_factory import LRSchedulerFactory

from easytext.component.register import ComponentRegister


@ComponentRegister.register(name_space="ner")
class LatticeLRSchedulerFactory(LRSchedulerFactory):
    """
    Lattice 模型的 LR scheduler
    """

    def __init__(self, gamma: float):
        super().__init__()
        self.gamma = gamma

    def create(self, optimizer: Optimizer, model: Model) -> "lr_scheduler":

        return ExponentialLR(optimizer=optimizer,
                             gamma=self.gamma)
