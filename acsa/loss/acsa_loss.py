#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
acsa loss

Authors: PanXu
Date:    2020/07/18 18:09:00
"""
import torch
from torch.nn import CrossEntropyLoss

from easytext.loss import Loss
from easytext.component.register import ComponentRegister
from acsa.models import ACSAModelOutputs


@ComponentRegister.register(name_space="acsa")
class ACSALoss(Loss):
    """
    ACSA Loss
    """

    def __init__(self):
        self.loss = CrossEntropyLoss()

    def __call__(self, model_outputs: ACSAModelOutputs, golden_label: torch.Tensor) -> torch.Tensor:

        return self.loss(model_outputs.logits, golden_label)
