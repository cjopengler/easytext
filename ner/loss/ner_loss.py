#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
ner loss

Authors: PanXu
Date:    2020/06/27 19:49:00
"""
import torch

from torch.nn import CrossEntropyLoss

from easytext.loss import Loss
from easytext.model import ModelOutputs
from easytext.modules import ConditionalRandomField
from easytext.component.register import ComponentRegister

from ner.models import NerModelOutputs
from ner.data.vocabulary_builder import VocabularyBuilder


@ComponentRegister.register(name_space="ner")
class NerLoss(Loss):
    """
    Ner CRF Loss
    """

    def __init__(self, vocabulary_builder: VocabularyBuilder):
        """
        loss 初始化
        :param vocabulary_builder: vocabulary builder
        """
        super().__init__()
        self.label_vocabulary = vocabulary_builder.label_vocabulary
        # 如果使用了 crf, 那么，则不使用该 loss
        self.loss = CrossEntropyLoss(ignore_index=self.label_vocabulary.padding_index)

    def __call__(self, model_outputs: ModelOutputs, golden_label: torch.Tensor) -> torch.Tensor:
        model_outputs: NerModelOutputs = model_outputs

        # shape: (batch_size, seq_len, label_size)
        logits = model_outputs.logits
        assert model_outputs.logits.dim() == 3, \
            f"model_outputs.logits.dim() != 3, 应该是 (batch_size, seq_len, label_size)"

        # shape: (batch_size, seq_len)
        mask = model_outputs.mask.long()
        assert mask.dim() == 2, f"mask.dim() != 2, 应该是 (batch_size, seq_len)"

        if model_outputs.crf is not None:
            crf: ConditionalRandomField = model_outputs.crf
            return -crf(inputs=logits,
                        tags=golden_label,
                        mask=mask) + (model_outputs.bert_pool * 0).sum()

        else:
            # 将 logits 转换成二维
            logits = logits.view(-1, self.label_vocabulary.label_size)
            golden_label = golden_label.view(-1)
            return self.loss(logits, golden_label)
