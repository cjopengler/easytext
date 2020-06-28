#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
ner scheduler factory

Authors: PanXu
Date:    2020/06/28 08:48:00
"""
from torch.optim import lr_scheduler
from torch.optim import Optimizer
from easytext.model import Model
from easytext.optimizer.lr_scheduler_factory import LRSchedulerFactory


class NerLRSchedulerFactory(LRSchedulerFactory):
    """
    Ner 模型的 LR scheduler
    """

    def __init__(self):
        super().__init__()

    def create(self, optimizer: Optimizer, model: Model) -> "lr_scheduler":
        pass
