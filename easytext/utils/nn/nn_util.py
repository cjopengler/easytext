#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2020/06/01 15:40:00
"""

import torch
from torch import LongTensor, BoolTensor


def has_tensor(obj) -> bool:
    """
    检查 对象中是否包含 Tensor
    :param obj:
    :return: True: 包含 Tensor; False: 没有 Tensor
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False


def sequence_mask(sequence: LongTensor, padding_index: int = 0) -> BoolTensor:
    """
    计算 sequence 序列的 mask
    :param sequence: sequence index, 维度是 (B, SeqLen)
    :param padding_index: padding 的index, 一般是 0，也可以根据自己的padding index 来设置。
    :return: sequence mask, 注意是 BoolTensor, 根据需要可以转化成 Float 或者 Long
    """

    if sequence.dim() != 2:
        raise RuntimeError(f"Sequence 的维度 {sequence.dim()} 不是 2，也就是 (B, SeqLen)")

    return sequence != padding_index

