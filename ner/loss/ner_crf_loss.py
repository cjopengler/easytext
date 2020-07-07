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

from ner.models import NerModelOutputs


class NerCRFLoss(Loss):
    """
    Ner CRF Loss
    """

    def __init__(self):
        super().__init__()

    def __call__(self, model_outputs: ModelOutputs, golden_label: torch.Tensor) -> torch.Tensor:
        model_outputs: NerModelOutputs = model_outputs

        crf: ConditionalRandomField = model_outputs.crf
        assert crf is not None, f"NerCRFLoss crf 不应该是 None"

        # shape: (batch_size, seq_len, label_size)
        logits = model_outputs.logits
        assert model_outputs.logits.dim() == 3, \
            f"model_outputs.logits.dim() != 3, 应该是 (batch_size, seq_len, label_size)"

        # shape: (batch_size, seq_len)
        mask = model_outputs.mask
        assert mask.dim() == 2, f"mask.dim() != 2, 应该是 (batch_size, seq_len)"

        return -crf(inputs=logits,
                    tags=golden_label,
                    mask=mask)
