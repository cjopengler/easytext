#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
acsa model outputs

Authors: PanXu
Date:    2020/07/18 16:00:00
"""

import torch
from easytext.component.register import ComponentRegister
from easytext.model import ModelOutputs


@ComponentRegister.register(name_space="acsa")
class ACSAModelOutputs(ModelOutputs):
    """
    ACSA Model Outputs
    """

    def __init__(self, logits: torch.FloatTensor):
        super().__init__(logits=logits)
