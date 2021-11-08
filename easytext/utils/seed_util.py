#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
设置随机数种子

Authors: PanXu
Date:    2021/11/07 12:17:00
"""

import torch
import numpy as np
import random


def set_seed(seed: int = 7) -> None:
    """
    设置相关函数的随机数种子
    :param seed: 随机数种子
    :return: None
    """

    # 随机数种子设定
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
    # 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
    torch.backends.cudnn.deterministic = True

    # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
    # 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
    torch.backends.cudnn.benchmark = False
