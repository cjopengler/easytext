#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
循环神经网络 sequence encoding 成 vector

Authors: PanXu
Date:    2020/10/20 17:19:00
"""

from typing import Dict

from torch.nn import Module
from torch import Tensor, BoolTensor

from easytext.modules import DynamicRnn, DynamicRnnOutput


class RnnSeq2Vec(Module):
    """
    rnn seq2vec, batch first
    """

    def __init__(self, dynamic_rnn: DynamicRnn):
        super().__init__()
        self.rnn = dynamic_rnn

    def forward(self, sequence: Tensor, mask: BoolTensor) -> Dict[str, Tensor]:
        rnn_output: DynamicRnnOutput = self.rnn(sequence=sequence,
                                                mask=mask)

        return rnn_output.last_layer_h_n
