#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
crf label index decoder

Authors: PanXu
Date:    2020/07/05 11:13:00
"""
from typing import List
import torch

from easytext.modules import ConditionalRandomField
from easytext.data import LabelVocabulary
from .label_index_decoder import LabelIndexDecoder


class CRFLabelIndexDecoder(LabelIndexDecoder):
    """
    CRF 的label index decoder, viterbi 算法
    """

    def __init__(self,
                 label_vocabulary: LabelVocabulary,
                 crf: ConditionalRandomField):
        self._crf = crf
        self._label_vocabulary = label_vocabulary

    def __call__(self,
                 logits: torch.Tensor,
                 mask: torch.ByteTensor) -> torch.LongTensor:

        assert logits.dim() == 3, \
            f"logits shape 不匹配, 应该是: (batch_size, seq_len, num_label), 现在是: {logits.dim()}"
        assert logits.size(-1) == self._label_vocabulary.label_size, \
            f"logits.size(-1) 与 label_size 不匹配, 现在是: {logits.size(-1)}"

        assert mask.dim() == 2, f"mask shape 不配, 应该是: (batch_size, seq_len), 现在是: {mask.dim()}"

        sequence_length = logits.size(1)

        best = self._crf.viterbi_tags(logits=logits, mask=mask)

        best_paths, scores = zip(*best)

        label_indices = list()

        for best_path in best_paths:
            best_path: List = best_path
            padding_indices = [self._label_vocabulary.padding_index] * (sequence_length - len(best_path))
            best_path.extend(padding_indices)

            label_indices.append(best_path)

        return torch.tensor(label_indices, dtype=torch.long)
