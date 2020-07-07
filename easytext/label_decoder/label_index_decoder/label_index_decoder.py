#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
label index decoder

Authors: PanXu
Date:    2020/07/05 11:12:00
"""

from typing import Tuple
import torch


class LabelIndexDecoder:
    """
    将 logits 解码成 label index
    """

    def __call__(self,
                 logits: torch.Tensor,
                 mask: torch.ByteTensor) -> torch.LongTensor:
        """
        将 logits 解码成 label index。
        例如:
        logits = [[0.3, 0.7], [0.8, 0.2]]
        返回的 index = [1, 0]

        :param logits: 模型输出 logits
        :param mask: mask
        :return: index
        """
        raise NotImplementedError()
