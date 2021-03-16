#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
tensor 的工具

Authors: PanXu
Date:    2021/02/15 12:32:00
"""

import torch


def is_tensor_equal(tensor1: torch.Tensor, tensor2: torch.Tensor, epsilon: float = 1e-9) -> bool:
    """
    判断两个 tensor 是否相等, 如果两个 tensor 的 size 不同，直接会抛出异常
    :param tensor1: 第一个
    :param tensor2:
    :param epsilon:
    :return: True: 相等; False: 不相等
    """

    assert tensor1.size() == tensor2.size(), f"tensor1 size: {tensor1.size()} 与 tensor2 size: {tensor2.size()} 不匹配"

    if tensor1.dtype == torch.long and tensor2.dtype == torch.long:
        # 都是整数的时候，需要将 epsilon 设置为 0
        epsilon = 0

    delta = tensor1 - tensor2

    delta_mask = delta.abs() > epsilon

    return delta_mask.sum() == 0
