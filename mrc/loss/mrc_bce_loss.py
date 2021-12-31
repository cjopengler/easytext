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

        start_loss = self.loss(model_outputs.start_logits.view(-1), golden_label["start_position_labels"].float().view(-1))
        # 计算得到 mean
        start_loss = (start_loss * mask.view(-1).float()).sum() / mask.sum()

        end_loss = self.loss(model_outputs.end_logits.view(-1), golden_label["end_position_labels"].float().view(-1))
        end_loss = (end_loss * mask.view(-1).float()).sum() / mask.sum()

        match_loss = self.loss(model_outputs.match_logits.view(batch_size, -1),
                               golden_label["match_position_labels"].float().view(batch_size, -1))

        match_label_row_mask = mask.bool().unsqueeze(-1).expand(-1, -1, sequence_length)
        match_label_col_mask = mask.bool().unsqueeze(-2).expand(-1, sequence_length, -1)
        match_label_mask = match_label_row_mask & match_label_col_mask

        match_label_mask = torch.triu(match_label_mask, 0)

        start_preds = model_outputs.start_logits > 0  # logits > 0, sigmoid > 0.5, 这是转换成标签0/1
        end_preds = model_outputs.end_logits > 0

        start_labels = golden_label["start_position_labels"]
        end_labels = golden_label["end_position_labels"]

        # start_preds.unsqueeze(-1).expand(-1, -1, seq_len) 产生 行 ,start 全部为1
        # & end_preds.unsqueeze(-2).expand(-1, seq_len, -1)) 产生 行 ,end 实际为 1 或 0
        # & 操作是的矩阵产生 span的位置上， start =1, end =1
        # logical_or 表示只要有 start 或 end 有一个与 golden 匹配上就是 1

        match_candidates = torch.logical_or(
            (start_preds.unsqueeze(-1).expand(-1, -1, sequence_length)
                & end_preds.unsqueeze(-2).expand(-1, sequence_length, -1)),
            (start_labels.unsqueeze(-1).expand(-1, -1, sequence_length)
                & end_labels.unsqueeze(-2).expand(-1, sequence_length, -1))
        )

        # 用来得到究竟有多少 span 预测对了
        match_label_mask = match_label_mask & match_candidates

        match_loss = match_loss * match_label_mask.view(batch_size, -1).float()
        match_loss = match_loss.sum() / (match_label_mask.sum() + 1e-10)

        mean_loss = start_loss * self.start_weight + end_loss * self.end_weight + match_loss * self.match_weight

        return mean_loss

