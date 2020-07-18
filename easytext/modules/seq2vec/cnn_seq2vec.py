#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
text cnn

Authors: PanXu
Date:    2020/07/07 20:49:00
"""
from typing import List, Union

import torch
from torch.nn import Module
from torch.nn import Conv1d
from torch.nn import ModuleList
from torch.nn import ReLU
from torch.nn import MaxPool1d


class CnnSeq2Vec(Module):
    """
    text cnn 编码器，将文本经过cnn
    完成的是 sequence 到 vector 转化
    """

    def __init__(self,
                 embedding_dim: int,
                 num_filters: int,
                 kernel_sizes: List[int]):
        """
        初始化
        :param embedding_dim: 输入的维度
        :param num_filters: filter 数量，也就是输出维度
        :param kernel_sizes: kernel size
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.cnn_layers = ModuleList([Conv1d(in_channels=embedding_dim,
                                             out_channels=num_filters,
                                             kernel_size=kernel_size) for kernel_size in kernel_sizes])

        self.activtion = ReLU()

    def forward(self, sequence: torch.FloatTensor, mask: Union[None, torch.ByteTensor]) -> torch.FloatTensor:
        """
        执行模型。对多个 kernel size 最终会将每一个 kernel 输出的向量, concat 在一起。
        pooling 使用的 max pooling.
        :param sequence: 输入的token 序列, shape: (batch_size, seq_len, embedding_dim)
        :param mask: mask
        :return: cnn 编码向量, shape: (batch_size, num_filter * len(kernel_sizes))
        """

        assert sequence.dim() == 3, f"tokens.dim: {sequence.dim()} 与 shape: (batch_size, seq_len, embedding_dim) 不匹配"

        if mask is not None:
            assert mask.dim() == 2, f"mask.dim: {mask.dim()} 与 shape: (batch_size, seq_len) 不匹配"

            # 将 mask 的 token 清零，避免影响 cnn
            sequence = sequence * mask.unsqueeze(dim=-1).float()

        # 将 1 和 2 转置, 转置后 shape: (batch_size, embedding_dim, seq_len)
        sequence = torch.transpose(sequence, 1, 2)

        # 每一个 cnn_vector_i: (batch_size, embedding_dim, new_seq_len_i)
        # 注意不同 kernel_size 的 cnn, 产生的 new_seq_len 长度是不同的 所以这里用下标 i 来表示.
        cnn_vectors = [self.activtion(cnn(sequence)) for cnn in self.cnn_layers]

        assert cnn_vectors[0].dim() == 3, \
            f"cnn_vectors.dim: {cnn_vectors[0].dim()} 与 shape: (batch_size, num_filter, new_seq_len) 不匹配"
        assert cnn_vectors[0].size(1) == self.num_filters

        # max pooling, 直接使用 max，而不是使用 MaxPool1D, max 更方便，MaxPool1D 需要设置 kernel size 为 seq_len.
        max_pooled_cnn_vectors = [cnn_vector.max(dim=-1)[0] for cnn_vector in cnn_vectors]
        assert max_pooled_cnn_vectors[0].dim() == 2, \
            f"max_pooled_cnn_vectors.dim: {max_pooled_cnn_vectors[0].dim()} 与 shape: (batch_size, num_filter) 不匹配"

        # 最后 max_pooled_cnn_vectors concat 在一起
        vector = \
            torch.cat(max_pooled_cnn_vectors, dim=-1) if len(max_pooled_cnn_vectors) > 1 else max_pooled_cnn_vectors[0]

        return vector




