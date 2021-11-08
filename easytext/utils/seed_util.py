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
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
