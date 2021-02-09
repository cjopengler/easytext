#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
lattice ner optimizer factory

Authors: PanXu
Date:    2021/02/09 11:25:00
"""

from torch.optim import SGD

from easytext.optimizer import OptimizerFactory
from easytext.model import Model
from easytext.component.register import ComponentRegister


@ComponentRegister.register(name_space="lattice")
class LatticeOptimizerFactory(OptimizerFactory):
    """
    Lattice Optimizer Factory
    """

    def __init__(self, lr: float, momentum: float):
        self.lr = lr
        self.momentum = momentum

    def create(self, model: Model) -> "LatticeOptimizerFactory":

        optimizer = SGD(params=model.parameters(),
                        lr=self.lr,
                        momentum=self.momentum)
        return optimizer


