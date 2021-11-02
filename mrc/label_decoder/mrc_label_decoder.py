#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
label decoder

Authors: PanXu
Date:    2021/11/02 09:18:00
"""
from typing import List

import torch

from easytext.label_decoder import LabelDecoder


class MRCLabelDecoder(LabelDecoder):
    """
    label decoder
    """

    def __call__(self, label_indices: torch.LongTensor, mask: torch.ByteTensor) -> List:
        """
        对 解码后的 label indices 解码出最终的 span label, 也就是 [begin, end] pair 的列表
        :param label_indices: 是 match label indices
        :param mask:
        :return: span label 列表, 每一行是一个样本中所有 [begin, end] pair. 结果就是:
                [[[begin00, end00], [begin01, end01], ...],
                 [[begin10, end10]],
                 ...
                 ]
        """

        batch_span_labels = list()  # 每一行是一个样本中得到的 [begin, end] pair
        for lable_indices in label_indices:
            span_labels = torch.nonzero(lable_indices.long()).tolist()
            batch_span_labels.append(span_labels)

        return batch_span_labels




