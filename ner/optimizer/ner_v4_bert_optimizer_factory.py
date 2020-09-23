#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
brief

Authors: PanXu
Date:    2020/09/10 21:10:00
"""
from transformers import AdamW

from easytext.optimizer import OptimizerFactory

from ner.models import NerV4


class NerV4BertOptimizerFactory(OptimizerFactory):
    """
    Ner Optimizer Factory 创建 Optimizer
    """

    def __init__(self, fine_tuning=False):
        self.fine_tuning = fine_tuning

    def create(self, model: NerV4) -> "NerOptimizerFactory":

        optimizer_grouped_parameters = list()

        # 增加 classifier 层参数
        optimizer_grouped_parameters.append(
            {
                "params": model.classifier.parameters(),
            }
        )

        # 增加 crf 参数
        if model.crf is not None:
            optimizer_grouped_parameters.append(
                {
                    "params": model.crf.parameters()
                }
            )

        if self.fine_tuning:
            # 设置 bert fine tune 参数
            no_decay = ['bias', 'LayerNorm.weight']
            bert_params = [
                {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            optimizer_grouped_parameters.extend(bert_params)

            optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5)
        else:
            bert_params = model.bert.parameters()
            for param in bert_params:
                param.requires_grad = False

            optimizer = AdamW(params=optimizer_grouped_parameters, lr=5e-5)

        return optimizer
