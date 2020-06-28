#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
模型输出

Authors: PanXu
Date:    2020/06/27 17:42:00
"""
import torch
from easytext.model import ModelOutputs


class NerModelOutputs(ModelOutputs):
    """
    Ner Model Outputs
    """

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor):

        super().__init__(logits=logits)
        self.mask = mask