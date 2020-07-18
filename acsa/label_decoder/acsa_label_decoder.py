#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
acsa label decoder

Authors: PanXu
Date:    2020/07/18 18:14:00
"""
from typing import List

import torch

from easytext.data import LabelVocabulary
from easytext.label_decoder import ModelLabelDecoder
from easytext.label_decoder import MaxLabelIndexDecoder

from acsa.models import ACSAModelOutputs


class ACSALabelDecoder(ModelLabelDecoder):
    """
    label decoder
    """

    def __init__(self, label_vocabulary: LabelVocabulary):
        self._label_index_decoder = MaxLabelIndexDecoder()
        self._label_vocabulary = label_vocabulary

    def decode_label_index(self, model_outputs: ACSAModelOutputs) -> torch.LongTensor:
        return self._label_index_decoder(logits=model_outputs.logits,
                                         mask=None)

    def decode_label(self, model_outputs: ACSAModelOutputs, label_indices: torch.LongTensor) -> List:
        return [self._label_vocabulary.token(index.item()) for index in label_indices]