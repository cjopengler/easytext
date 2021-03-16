#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
循环神经网络 rnn sequence 2 sequence

Authors: PanXu
Date:    2020/10/20 17:24:00
"""

from typing import Dict

from torch.nn import Module
from torch import Tensor, BoolTensor

from easytext.modules import DynamicRnn, DynamicRnnOutput


class RnnSeq2Seq(Module):
    """
    rnn seq2vec, batch first
    """

    def __init__(self, dynamic_rnn: DynamicRnn):
        super().__init__()
        self.rnn = dynamic_rnn

    def forward(self, sequence: Tensor, mask: BoolTensor) -> Tensor:
        """
        基于 rnn 的 seq2seq encoder
        :param sequence: sequence embedding, shape: (B, seq_len, embedding_size)
        :param mask: bool mask, shape: (B, seq_len)
        :return: seq2seq 解码结果, shape: (B, seq_len, hidden_size)
        """
        rnn_output: DynamicRnnOutput = self.rnn(sequence=sequence, mask=mask)

        return rnn_output.output
