#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
mrc model label decoder

Authors: PanXu
Date:    2021/10/27 16:01:00
"""
from typing import List

import torch

from easytext.label_decoder import ModelLabelDecoder
from easytext.label_decoder import CRFLabelIndexDecoder
from easytext.label_decoder import SequenceLabelDecoder
from easytext.component.register import ComponentRegister

from mrc.models import MRCNerOutput

from mrc.label_decoder import MRCLabelIndexDecoder
from mrc.label_decoder import MRCLabelDecoder


@ComponentRegister.register(name_space="mrc")
class MRCModelLabelDecoder(ModelLabelDecoder):
    """
    MRC Model Label Decoder
    """

    def __init__(self):
        super().__init__()
        self._label_index_decoder = MRCLabelIndexDecoder()
        self._label_decoder = MRCLabelDecoder()

    def decode_label_index(self, model_outputs: MRCNerOutput) -> torch.LongTensor:
        """
        解码 label index
        :param model_outputs: 模型输出结果
        :return: label indices， 注意 device 是 cpu 的。
        """
        start_logits = model_outputs.start_logits.detach()
        end_logits = model_outputs.end_logits.detach()
        match_logits = model_outputs.match_logits.detach()
        mask = model_outputs.mask.detach()

        return self._label_index_decoder(start_logits=start_logits,
                                         end_logits=end_logits,
                                         match_logits=match_logits,
                                         mask=mask)

    def decode_label(self, model_outputs: MRCNerOutput, label_indices: torch.LongTensor) -> List:
        """
        将 label indices 解码成 span list
        :param model_outputs: 模型输出
        :param label_indices: label indices
        :return: span list
        """

        mask = model_outputs.mask.detach().cpu()
        label_indices = label_indices.detach().cpu()
        return self._label_decoder(label_indices=label_indices, mask=mask)

