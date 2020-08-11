#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
基于 attention 的 seq2vec

Authors: PanXu
Date:    2020/07/18 12:00:00
"""
from typing import Optional

import torch
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Tanh

from easytext.utils.nn.nn_util import masked_softmax


class AttentionSeq2Vec(Module):
    """
    基于 attention 将 seq2vec. 具体操作如下:

    1. sequence: (B, seq_len, input_size)
    2. K = WkSeqeunce 将 sequence 进行变换, K shape: (B, seq_len, query_hidden_size)
    3. Q = Shape: (query_hidden_size)
    4. attention = softmax(KQ), shape: (B, seq_len)
    5. V = WvSequence, shape: (B, seq_len, value_hidden_size); 如果 value_hidden_size is None,
    shape: (B, seq_len, input_size)
    6. sum(V*attention, dim=-1), shape: (B, input_size)
    """

    def __init__(self,
                 input_size: int,
                 query_hidden_size: int,
                 value_hidden_size: Optional[int] = None):
        """
        初始化。遵循 Q K V，计算 attention 方式。
        :param input_size: 输入的 sequence token 的 embedding dim
        :param query_hidden_size: 将 seqence 变成 Q 的时候，变换后的 token embedding dim.
        :param value_hidden_size: 将 seqence 变成 V 的时候, 变换后的 token embedding dim.
        如果 value_hidden_size is None, 那么，该模型就与 2016-Hierarchical Attention Networks for Document Classification
        是一致的, 最后的输出结果 shape (B, seq_len, input_size);
        如果 value_hidden_size 被设置了, 那么，就与 Attention is All your Need 中 变换是一致的, 最后的输出结果
        shape (B, seq_len, value_hidden_size)
        """
        super().__init__()
        # 对 seqence 转化成 K
        self.wk = Linear(in_features=input_size, out_features=query_hidden_size, bias=True)
        self.key_activation = Tanh()

        # attention 计算 dot 相似度，所以 bias = False
        self.attention = Linear(in_features=query_hidden_size, out_features=1, bias=False)

        self.wv = None
        if value_hidden_size is not None:
            self.wv = Linear(in_features=input_size, out_features=value_hidden_size, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, sequence: torch.LongTensor, mask: Optional[torch.ByteTensor]) -> torch.FloatTensor:
        """
        执行 attetion seq2vec
        :param sequence: 输入的token 序列, shape: (batch_size, seq_len, input_size)
        :param mask: mask shape: (batch_size, seq_len)
        :return: attention 编码向量, shape: (batch_size, value_hidden_size or input_size)
        """

        assert sequence.dim() == 3, f"sequence shape: (batch_size, seq_len, input_size)"

        if mask is not None:
            assert mask.dim() == 2, f"mask shape: (batch_size, seq_len)"

        # sequence 转换成 key, key shape: (B, seq_len, query_ebmedding_size)
        key = self.wk(sequence)
        key = self.key_activation(key)

        # Q * K 计算相似度, attention shape: (B, seq_len)
        attention = self.attention(key)
        attention = torch.squeeze(attention, dim=-1)

        # 对 attention 归一化
        if mask is not None:
            attention = masked_softmax(vector=attention, mask=mask)
        else:
            attention = torch.softmax(attention, dim=-1)

        # 计算value
        if self.wv is not None:
            # value shape: (B, seq_len, value_hidden_size)
            value = self.wv(sequence)
        else:
            # value shape: (B, seq_len, embedding_dim)
            value = sequence

        # attentioned_value shape: (B, seq_len, value_hidden_size or embedding_dim)
        attentioned_value = value * attention.unsqueeze(dim=-1)

        # 将 attenctioned value 相加在一起, vector shape: (B, value_hidden_size or embedding_dim)
        vector = torch.sum(attentioned_value, dim=1)
        return vector

