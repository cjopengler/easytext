#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
optimizer factory

Authors: PanXu
Date:    2020/06/28 08:42:00
"""

from torch.optim import Adam

from easytext.optimizer import OptimizerFactory
from easytext.model import Model
from easytext.component.register import ComponentRegister


@ComponentRegister.register(name_space="ner")
class NerOptimizerFactory(OptimizerFactory):
    """
    Ner Optimizer Factory 创建 Optimizer
    """

    def __init__(self, fine_tuning=False):
        self.fine_tuning = fine_tuning

    def create(self, model: Model) -> "NerOptimizerFactory":

        optimizer = Adam(params=model.parameters(),
                         lr=0.01)
        return optimizer
