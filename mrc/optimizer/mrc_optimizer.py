#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
优化器

Authors: PanXu
Date:    2021/11/05 08:53:00
"""
from easytext.model import Model
from easytext.optimizer import OptimizerFactory
from easytext.component.register import ComponentRegister

from transformers import AdamW


@ComponentRegister.register(name_space="mrc_ner")
class MRCOptimizer(OptimizerFactory):

    def __init__(self, lr: float, eps: float, weight_decay: float):
        self.lr = lr
        self.eps = eps
        self.weight_decay = weight_decay

    def create(self, model: Model) -> "Optimizer":
        no_decay = ["bias", "LayerNorm.weight"]
        parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]

        optimizer = AdamW(parameters,
                          betas=(0.9, 0.98),  # RoBERTa paper 参数
                          lr=self.lr,
                          eps=self.eps)

        return optimizer
