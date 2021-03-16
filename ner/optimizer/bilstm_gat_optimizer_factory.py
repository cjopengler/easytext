#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
bilstm gat optimizer factory

Authors: PanXu
Date:    2021/02/18 17:07:00
"""

from torch.optim import Adam, SGD

from easytext.optimizer import OptimizerFactory
from easytext.model import Model
from easytext.component.register import ComponentRegister


@ComponentRegister.register(name_space="ner")
class BilstmGATOptimizerFactory(OptimizerFactory):
    """
    bilstm gat Optimizer Factory
    """

    def __init__(self, lr: float, weight_decay: float, optimizer_name: str = "Adam"):
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name

        assert optimizer_name in {"Adam", "SGD"}, f"optimizer_name 必须是 Adam 或 SGD"

    def create(self, model: Model) -> "BilstmGATOptimizerFactory":

        if self.optimizer_name == "SGD":
            optimizer = SGD(params=model.parameters(),
                            lr=self.lr,
                            weight_decay=self.weight_decay)
        elif self.optimizer_name == "Adam":
            optimizer = Adam(params=model.parameters(),
                             lr=self.lr,
                             weight_decay=self.weight_decay)
        else:
            raise RuntimeError(f"optimizer_name 必须是 Adam 或 SGD")
        return optimizer

