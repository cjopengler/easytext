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
from easytext.modules import ConditionalRandomField


class NerModelOutputs(ModelOutputs):
    """
    Ner Model Outputs
    """

    def __init__(self,
                 logits: torch.Tensor,
                 mask: torch.Tensor,
                 crf: ConditionalRandomField = None,
                 bert_pooler_output: torch.Tensor = None):
        """
        Ner 模型的输出
        :param logits: logits 输出
        :param mask: mask
        :param crf: 模型中的 crf 输出出来，用来进行 loss 以及 viterbi 解码
        :param bert_pooler_output: bert pooler output
        """

        super().__init__(logits=logits)
        self.mask = mask
        self.crf = crf
        self.bert_pooler_output = bert_pooler_output


