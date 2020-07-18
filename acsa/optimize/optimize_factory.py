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

from easytext.model import Model
from easytext.optimizer import OptimizerFactory


class ACSAOptimizerFactory(OptimizerFactory):

    def __init__(self, config: Dict):
        self.config = config

    def create(self, model: Model) -> "Optimizer":

        return Adam(params=model.parameters(),
                    lr=0.01)