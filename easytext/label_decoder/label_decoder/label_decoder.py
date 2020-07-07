#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
label decoder

Authors: PanXu
Date:    2020/07/05 10:38:00
"""
from typing import List
import torch


class LabelDecoder:
    """
    将 label index 转化成 label，最终用户需要的结果
    """

    def __call__(self, label_indices: torch.LongTensor, mask: torch.ByteTensor) -> List:
        raise NotImplementedError()
