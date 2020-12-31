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
from torch.optim import Adam
from transformers import AdamW

from easytext.optimizer import OptimizerFactory

from ner.models import BertRnnWithCrf

from easytext.component.register import ComponentRegister


@ComponentRegister.register(name_space="ner")
class BertRnnWithCrfOptimizerFactory(OptimizerFactory):
    """
    Ner Optimizer Factory 创建 Optimizer
    """

    def __init__(self, fine_tuning=False):
        self.fine_tuning = fine_tuning

    def create(self, model: BertRnnWithCrf) -> "BertRnnWithCrfOptimizerFactory":

        optimizer_grouped_parameters = list()

        # 增加 linear 参数
        optimizer_grouped_parameters.append({
                "params": model.liner.parameters(),
                "lr": 0.01
            }
        )

        # 增加 rnn 参数
        optimizer_grouped_parameters.append({
            "params": model.rnn_seq2seq.parameters()
            }
        )

        # 增加 crf 参数
        if model.crf is not None:
            optimizer_grouped_parameters.append({
                "params": model.crf.parameters()
                }
            )

        if self.fine_tuning:
            # 设置 bert fine tune 参数
            no_decay = ['bias', 'LayerNorm.weight']
            bert_params = [
                {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01,
                 "lr": 5e-5},
                {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0,
                 "lr": 5e-5}
            ]

            optimizer_grouped_parameters.extend(bert_params)

            optimizer = AdamW(optimizer_grouped_parameters, lr=0.01)
        else:
            bert_params = model.bert.parameters()
            for param in bert_params:
                param.requires_grad = False

            optimizer = Adam(params=optimizer_grouped_parameters, lr=0.01)

        return optimizer


