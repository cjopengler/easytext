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

from ner.models import NerModelOutputs


class NerLoss(Loss):
    """
    Ner loss
    """

    def __init__(self):
        super().__init__()
        self.loss = CrossEntropyLoss()

    def __call__(self, model_outputs: ModelOutputs, golden_label: torch.Tensor) -> torch.Tensor:
        bool_mask = (model_outputs.mask != 0)

        model_outputs: NerModelOutputs = model_outputs

        # shape: (batch_size, seq_len, label_size)
        assert model_outputs.logits.dim() == 3, \
            f"model_outputs.logits.dim() != 3, 应该是 (batch_size, seq_len, label_size)"

        # shape: (batch_size, seq_len, label_size)
        logits = model_outputs.logits
        label_size = logits.shape[-1]

        # shape: (batch_size, dim), 将masked的去除掉，剩下的就是需要的
        logits_flat = torch.masked_select(logits, bool_mask.unsqueeze(-1)).contiguous().view(-1, label_size)

        # shape: (batch_size, seq_len)
        assert model_outputs.mask.dim() == 2, \
            f"mask shape 应该是 (batch_size, seq_len), 现在是: {model_outputs.mask.dim()}"

        # shape: (batch_size, seq_len)
        assert golden_label.dim() == 2

        # golden label flat shape: (B,)
        golden_label_flat = torch.masked_select(golden_label, bool_mask)

        assert golden_label_flat.dim() == 1

        return self.loss(logits_flat, golden_label_flat)

