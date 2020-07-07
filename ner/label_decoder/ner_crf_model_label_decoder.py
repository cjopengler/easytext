#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
crf model label decoder, use viterbi

Authors: PanXu
Date:    2020/07/07 09:35:00
"""
from typing import List

import torch

from easytext.label_decoder import ModelLabelDecoder
from easytext.label_decoder import CRFLabelIndexDecoder
from easytext.label_decoder import SequenceLabelDecoder
from easytext.data import LabelVocabulary

from ner.models import NerModelOutputs


class NerCRFModelLabelDecoder(ModelLabelDecoder):
    """
    Ner CRF Model Label Decoder
    """

    def __init__(self, label_vocabulary: LabelVocabulary):
        super().__init__()
        self._label_index_decoder = None
        self._label_decoder = None
        self._label_vocabulary = label_vocabulary

    def decode_label_index(self, model_outputs: NerModelOutputs) -> torch.LongTensor:
        """
        解码出 label index, 使用 crf 解码
        :param model_outputs: 模型输出结果
        :return: label indices
        """

        if self._label_index_decoder is None:
            self._label_index_decoder = CRFLabelIndexDecoder(label_vocabulary=self._label_vocabulary,
                                                             crf=model_outputs.crf)

        return self._label_index_decoder(logits=model_outputs.logits,
                                         mask=model_outputs.mask)

    def decode_label(self, model_outputs: NerModelOutputs, label_indices: torch.LongTensor) -> List:
        """
        将 label indices 解码成 span list
        :param model_outputs: 模型输出
        :param label_indices: label indices
        :return: span list
        """
        if self._label_decoder is None:
            self._label_decoder = SequenceLabelDecoder(label_vocabulary=self._label_vocabulary)

        return self._label_decoder(label_indices=label_indices, mask=model_outputs.mask)

