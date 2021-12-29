#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
label index decoder

Authors: PanXu
Date:    2021/11/02 08:45:00
"""
import torch

from easytext.label_decoder import LabelIndexDecoder

from mrc.models import MRCNerOutput


class MRCLabelIndexDecoder(LabelIndexDecoder):
    """
    MRC label index decoder
    """

    def __call__(self,
                 start_logits: torch.Tensor,
                 end_logits: torch.Tensor,
                 match_logits: torch.Tensor,
                 mask: torch.BoolTensor) -> torch.LongTensor:

        mask = mask.bool()
        batch_size, seq_len = start_logits.size()

        # match label pred, [batch_size, seq_len, seq_len]
        match_preds = match_logits > 0

        # mask 保留 match_preds 或者 start, end 其中之一即可
        match_preds = match_preds \
                      & mask.unsqueeze(-1).expand(-1, -1, seq_len) \
                      & mask.unsqueeze(1).expand(-1, seq_len, -1)

        # [batch_size, seq_len]
        start_preds = start_logits > 0

        start_preds = start_preds & mask

        # [batch_size, seq_len]
        end_preds = end_logits > 0
        end_preds = end_preds & mask

        # match label 最终结果
        match_preds = (match_preds
                       & start_preds.unsqueeze(-1).expand(-1, -1, seq_len)
                       & end_preds.unsqueeze(1).expand(-1, seq_len, -1))

        return match_preds
