#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
ner max label decoder

Authors: PanXu
Date:    2020/07/06 12:39:00
"""
from typing import List

import torch

from easytext.data import LabelVocabulary

from easytext.label_decoder import ModelLabelDecoder
from easytext.label_decoder import SequenceMaxLabelIndexDecoder
from easytext.label_decoder import SequenceLabelDecoder
from easytext.component.register import ComponentRegister
from ner.models import NerModelOutputs


@ComponentRegister.register_class(name="NerMaxModelLabelDecoder", name_space="label_decoder")
class NerMaxModelLabelDecoder(ModelLabelDecoder):

    def __init__(self, label_vocabulary: LabelVocabulary):
        super().__init__()
        self._decode_label_index = SequenceMaxLabelIndexDecoder(label_vocabulary=label_vocabulary)
        self._label_decoder = SequenceLabelDecoder(label_vocabulary=label_vocabulary)

    def decode_label_index(self, model_outputs: NerModelOutputs) -> torch.LongTensor:
        logits = model_outputs.logits.detach().cpu()
        mask = model_outputs.mask.detach().cpu()

        return self._decode_label_index(logits=logits, mask=mask)

    def decode_label(self, model_outputs: NerModelOutputs, label_indices: torch.LongTensor) -> List:
        mask = model_outputs.mask.detach().cpu()
        return self._label_decoder(label_indices=label_indices, mask=mask)