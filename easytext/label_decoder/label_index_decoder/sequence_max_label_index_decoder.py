#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
通过取到最大值的 sequence label 解码

Authors: PanXu
Date:    2020/07/05 11:28:00
"""

from typing import Tuple
import torch

from easytext.utils import bio as BIO
from easytext.data import LabelVocabulary
from .label_index_decoder import LabelIndexDecoder


class SequenceMaxLabelIndexDecoder(LabelIndexDecoder):
    """
    对于 sequence logits, shape: (batch_size, seq_len, num_label), 使用 max 进行 在每一个
    timestep 上进行 decode, 得到 label index.
    """

    def __init__(self, label_vocabulary: LabelVocabulary):
        """
        初始化
        :param label_vocabulary: label 词汇表
        """
        self._label_vocabulary = label_vocabulary

    def __call__(self, logits: torch.Tensor, mask: torch.ByteTensor) -> torch.LongTensor:
        """
        对于 sequence logits, shape: (batch_size, seq_len, num_label), 使用 max 进行 在每一个
        timestep 上进行 decode, 得到 label index.
        :param logits: shape: (batch_size, seq_len, num_label)
        :param mask: shape: (bath_size, seq_len), 存储的是 0 或 1
        :return: 解码后的 label index, shape: (batch_size, seq_len), 注意这是有padding_index 的结果，
        需要使用 mask 来提取实际的 label index.
        """
        if logits.dim() != 3:
            raise RuntimeError(f"logits shape 错误, 应该是 (B, seq_len, num_label), "
                               f"而现在是 {logits.shape}")

        if (mask is not None) and (mask.dim() != 2):
            raise RuntimeError(f"mask shape 错误, 应该是 (B, seq_len), "
                               f"而现在是 {mask.shape}")

        batch = logits.size(0)
        max_sequence_length = logits.size(1)

        # mask shape: (B, seq_len)
        if mask is None:
            mask = torch.ones(size=(logits.shape[0], logits.shape[1]),
                              dtype=torch.long)

        sequence_length = mask.sum(dim=-1).tolist()

        batch_indices = list()
        for i in range(batch):
            sequence_labels, sequence_label_indices = BIO.decode_one_sequence_logits_to_label(
                sequence_logits=logits[i, :sequence_length[i]],
                vocabulary=self._label_vocabulary)

            padding_indices = [self._label_vocabulary.padding_index] * (max_sequence_length - sequence_length[i])
            sequence_label_indices.extend(padding_indices)

            batch_indices.append(sequence_label_indices)

        batch_indices = torch.tensor(batch_indices, dtype=torch.long)
        return batch_indices
