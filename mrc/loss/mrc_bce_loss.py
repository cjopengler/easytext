#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
mrc bce loss

Authors: PanXu
Date:    2021/10/27 14:18:00
"""

from typing import Dict

import torch
from torch.nn import BCEWithLogitsLoss

from mrc.models import MRCNerOutput

from easytext.component.register import ComponentRegister


@ComponentRegister.register(name_space="mrc_ner")
class MRCBCELoss:
    """
    基于 bce 的 los
    """

    def __init__(self, start_weight: float, end_weight: float, match_weight: float):
        weight_sum = start_weight + end_weight + match_weight
        self.start_weight = start_weight / weight_sum
        self.end_weight = end_weight / weight_sum
        self.match_weight = match_weight / weight_sum

        self.loss = BCEWithLogitsLoss(reduction="none")

    def __call__(self, model_outputs: MRCNerOutput, golden_label: Dict[str, torch.Tensor]):
        mask = model_outputs.mask.long()

        batch_size, sequence_length = model_outputs.start_logits.size()

        start_loss = self.loss(model_outputs.start_logits, golden_label["start_position_labels"])
        # 计算得到 mean
        start_loss = (start_loss * mask).sum() / mask.sum()

        end_loss = self.loss(model_outputs.end_logits, golden_label["end_position_labels"])
        end_loss = (end_loss * mask).sum() / mask.sum()

        match_loss = self.loss(model_outputs.match_logits.view(batch_size, -1),
                               golden_label["match_position_labels"].view(batch_size, -1))

        match_label_row_mask = mask.bool().unsqueeze(-1).expand(-1, -1, sequence_length)
        match_label_col_mask = mask.bool().unsqueeze(-2).expand(-1, sequence_length, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask

        match_label_mask = torch.triu(match_label_mask, 0)
        match_label_mask = match_label_mask.view(batch_size, -1)

        match_loss = match_loss * match_label_mask
        match_loss = match_loss.sum() / (match_label_mask.sum() + 1e-10)

        total_loss = start_loss * self.start_weight + end_loss * self.end_weight + match_loss * self.match_weight
        mean_loss = total_loss / batch_size

        return mean_loss

