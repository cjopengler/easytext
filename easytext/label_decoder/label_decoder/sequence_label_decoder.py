#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
sequence label decoder

Authors: PanXu
Date:    2020/07/05 11:27:00
"""
from typing import List

import torch

from easytext.data import LabelVocabulary
from easytext.utils import bio as BIO

from .label_decoder import LabelDecoder


class SequenceLabelDecoder(LabelDecoder):
    """
    对序列label解码成用户需要的
    """

    def __init__(self, label_vocabulary: LabelVocabulary):
        self._label_vocabulary = label_vocabulary

    def __call__(self, label_indices: torch.LongTensor, mask: torch.ByteTensor) -> List:
        """
        将 label index 解码 成span

        batch_sequence_label shape:(B, seq_len)  (B-T: 0, I-T: 1, O: 2)
        label index:
        [[0, 1, 2],
         [2, 0, 1]]

         对应label序列是:
         [[B, I, O],
          [O, B, I]]

         解码成:

         [[{"label": T, "begin": 0, "end": 2}],
          [{"label": T, "begin": 1, "end": 3}]]

        :param label_indices: label index
        :param mask: mask
        :return: span list
        """
        spans = BIO.decode_label_index_to_span(
            batch_sequence_label_index=label_indices,
            mask=mask,
            vocabulary=self._label_vocabulary)

        return spans
