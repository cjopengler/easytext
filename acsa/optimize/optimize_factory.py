#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
optimize factory

Authors: PanXu
Date:    2020/07/18 23:35:00
"""
from typing import Dict

from torch.optim import Adam
from easytext.component.register import ComponentRegister
from easytext.model import Model
from easytext.optimizer import OptimizerFactory


@ComponentRegister.register(name_space="acsa")
class ACSAOptimizerFactory(OptimizerFactory):
    """
    ACSA 优化器
    """

    def __init__(self, fine_tuning=False):
        self.fine_tuning = fine_tuning

    def create(self, model: Model) -> "Optimizer":
        return Adam(params=model.parameters(),
                    lr=0.01)
