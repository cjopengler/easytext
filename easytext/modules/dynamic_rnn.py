#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
动态 rnn 处理变长序列，batch first

Authors: PanXu
Date:    2020/10/20 15:28:00
"""

from typing import Dict

import torch
from torch import Tensor, BoolTensor
from torch.nn import Module, LSTM, GRU, RNN, RNNBase
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class DynamicRnnOutput:
    """
    成员变量:
    * last_layer_h_n: 最后一层的最后一个 time step 隐层输出，
                      batch first shape: (batch, hidden_size * num_directions)
    * last_layer_c_n: 最后一层的最后一个 time step cell输出
                      batch first shape: (batch, hidden_size * num_directions)
    * h_n: 所有层的最后一个 time step 的隐层输出，具体参考 pytorch lstm/gru/rnn 文档描述
                      batch first shape: (batch, num_layers * num_directions, hidden_size)
    * c_n: 所有层的最后一个 time step 的cell 输出, 具体参考 pytorch lstm/gru/rnn 文档描述
           对于 rnn 来说，为 None. batch first shape: (batch, num_layers * num_directions, hidden_size)
    * output: 经过 rnn 后的序列输出, 具体参考 pytorch lstm/gru/rnn 文档描述
              batch first shape: (batch, seq_len, num_directions * hidden_size),
    """

    def __init__(self,
                 last_layer_h_n: Tensor,
                 last_layer_c_n: Tensor,
                 h_n: Tensor,
                 c_n: Tensor,
                 sequence_encoding: Tensor):
        """
        :param last_layer_h_n: 最后一层的最后一个 time step 隐层输出，
                batch first shape: (batch, hidden_size * num_directions)
        :param last_layer_c_n: 最后一层的最后一个 time step cell输出
                batch first shape: (batch, hidden_size * num_directions)
        :param h_n: 所有层的最后一个 time step 的隐层输出，具体参考 pytorch lstm/gru/rnn 文档描述
                batch first shape: (batch, num_layers * num_directions, hidden_size)
        :param c_n: 所有层的最后一个 time step 的cell 输出, 具体参考 pytorch lstm/gru/rnn 文档描述
                对于 rnn 来说，为 None. batch first shape: (batch, num_layers * num_directions, hidden_size)
        :param sequence_encoding: 经过 rnn 后的序列输出, 具体参考 pytorch lstm/gru/rnn 文档描述
                batch first shape: (batch, seq_len, num_directions * hidden_size),
        """
        self.last_layer_h_n = last_layer_h_n
        self.last_layer_c_n = last_layer_c_n
        self.h_n = h_n
        self.c_n = c_n
        self.output = sequence_encoding


class DynamicRnn(Module):
    """
    动态 rnn, 注意所有的都是 batch first。
    """
    LSTM = "lstm"
    GRU = "gru"
    RNN = "rnn"

    def __init__(self, rnn: RNNBase):
        super().__init__()

        self.rnn = rnn

        if isinstance(self.rnn, LSTM):
            self.rnn_type = DynamicRnn.LSTM
        elif isinstance(self.rnn, GRU):
            self.rnn_type = DynamicRnn.GRU
        elif isinstance(self.rnn, RNN):
            self.rnn_type = DynamicRnn.RNN
        else:
            raise RuntimeError(f"rnn 类型: {type(self.rnn)} 不属于 "
                               f"{DynamicRnn.LSTM}, {DynamicRnn.GRU}, {DynamicRnn.RNN}")

        assert self.rnn.batch_first, f"rnn batch_first 必须设置为 True"

        self.num_layers = self.rnn.num_layers

    def forward(self, sequence: Tensor, mask: BoolTensor) -> DynamicRnnOutput:
        """
        rnn 执行。特别注意: 所有的都是 batch first
        :param sequence: sequence 序列, shape: (B, seq_len, input_size)
        :param mask: 对 sequence 的 mask, shape: (B, seq_len)
        :return: 解码后的结果，具体参考 DynamicOutput 说明
        """
        assert sequence.dim() == 3, \
            f"sequence shape: {sequence.dim()} 与 (B, seq_len, input_size) 不匹配"

        assert sequence.size(-1) == self.rnn.input_size, \
            f"sequence.size(-1): {sequence.size(-1)} 与 rnn input_size: {self.rnn.input_size} 不相等"

        batch_size = sequence.size(0)
        sequence_length = sequence.size(1)

        sequence_lengths = mask.sum(dim=-1)

        pack = pack_padded_sequence(sequence,
                                    lengths=sequence_lengths,
                                    batch_first=True,
                                    enforce_sorted=False)

        packed_sequence_encoding, last_state = self.rnn(pack)

        encoding, pad_sequence_length = pad_packed_sequence(packed_sequence_encoding,
                                                            batch_first=True,
                                                            padding_value=0.0,
                                                            total_length=sequence_length)

        if self.rnn_type == DynamicRnn.LSTM or self.rnn_type == DynamicRnn.GRU:
            h_n, c_n = last_state
        else:
            h_n = last_state
            c_n = None

        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # 因为是按照 batch first 来处理的，所以需要进行转换
        # 转换之后的 h_n shape: (batch, num_layers, hidden_size * num_directions), c_n 同样的处理
        h_n = torch.transpose(h_n, 0, 1).contiguous().view(batch_size, self.num_layers, -1)
        last_layer_h_n = h_n[:, -1, :].contiguous().view(batch_size, -1)

        last_layer_c_n = None
        if c_n is not None:
            c_n = torch.transpose(c_n, 0, 1).contiguous().view(batch_size, self.num_layers, -1)
            last_layer_c_n = c_n[:, -1, :].contiguous().view(batch_size, -1)

        return DynamicRnnOutput(last_layer_h_n=last_layer_h_n,
                                last_layer_c_n=last_layer_c_n,
                                h_n=h_n,
                                c_n=c_n,
                                sequence_encoding=encoding)
