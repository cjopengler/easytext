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
from easytext.component.register import ComponentRegister
from acsa.models import ACSAModelOutputs
from acsa.data.vocabulary_builder import VocabularyBuilder


@ComponentRegister.register(name_space="acsa")
class ACSALabelDecoder(ModelLabelDecoder):
    """
    label decoder
    """

    def __init__(self, vocabulary_builder: VocabularyBuilder):
        self._label_index_decoder = MaxLabelIndexDecoder()
        self._label_vocabulary = vocabulary_builder.label_vocabulary

    def decode_label_index(self, model_outputs: ACSAModelOutputs) -> torch.LongTensor:
        return self._label_index_decoder(logits=model_outputs.logits,
                                         mask=None)

    def decode_label(self, model_outputs: ACSAModelOutputs, label_indices: torch.LongTensor) -> List:
        return [self._label_vocabulary.token(index.item()) for index in label_indices]