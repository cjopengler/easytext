#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
max label decoder

Authors: PanXu
Date:    2020/07/06 10:13:00
"""
from typing import Union
import torch

from .label_index_decoder import LabelIndexDecoder


class MaxLabelIndexDecoder(LabelIndexDecoder):
    """
    max label index decoder
    """

    def __call__(self, logits: torch.Tensor, mask: Union[None, torch.ByteTensor] = None) -> torch.LongTensor:
        """
        基于 argmax 从 logits 中提取出 max index
        :param logits: logits shape: (batch_size, num_label)
        :param mask: mask 必须是 None
        :return:
        """

        if mask is not None:
            raise RuntimeError(f"mask 必须是 None")

        return torch.argmax(logits, dim=-1)

