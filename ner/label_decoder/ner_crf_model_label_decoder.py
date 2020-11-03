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
from easytext.component.register import ComponentRegister

from ner.models import NerModelOutputs
from ner.data.vocabulary_builder import VocabularyBuilder


@ComponentRegister.register(name_space="ner")
class NerCRFModelLabelDecoder(ModelLabelDecoder):
    """
    Ner CRF Model Label Decoder
    """

    def __init__(self, vocabulary_builder: VocabularyBuilder):
        super().__init__()
        self._label_index_decoder = None
        self._label_decoder = None
        self._label_vocabulary = vocabulary_builder.label_vocabulary

    def decode_label_index(self, model_outputs: NerModelOutputs) -> torch.LongTensor:
        """
        解码出 label index, 使用 crf 解码
        :param model_outputs: 模型输出结果
        :return: label indices， 注意 device 是 cpu 的。
        """

        # 不转换成 cpu, 因为在 CRFLabelIndexDecoder 用到了 viterbi
        # 而 viterbi 用到了 crf 的 transition matrix 参数
        # transition matrix 参数，是模型的一部分。所以如果模型是 gpu，那么
        # transition matrix 就是 GPU. 这样保证 logits, mask 与 transition matrix 的 device 一致。
        logits = model_outputs.logits.detach()
        mask = model_outputs.mask.detach()

        if self._label_index_decoder is None:
            self._label_index_decoder = CRFLabelIndexDecoder(label_vocabulary=self._label_vocabulary,
                                                             crf=model_outputs.crf)

        return self._label_index_decoder(logits=logits,
                                         mask=mask)

    def decode_label(self, model_outputs: NerModelOutputs, label_indices: torch.LongTensor) -> List:
        """
        将 label indices 解码成 span list
        :param model_outputs: 模型输出
        :param label_indices: label indices
        :return: span list
        """
        if self._label_decoder is None:
            self._label_decoder = SequenceLabelDecoder(label_vocabulary=self._label_vocabulary)

        mask = model_outputs.mask.detach().cpu()
        label_indices = label_indices.detach().cpu()
        return self._label_decoder(label_indices=label_indices, mask=mask)

