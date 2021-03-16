#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
图注意力模型

Authors: PanXu
Date:    2021/02/14 10:00:00
"""

import torch
from torch import nn
from torch.nn import Dropout, Parameter, Linear, LeakyReLU, ModuleList
from torch.nn import ELU, LogSoftmax
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    图 Attention Layer
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout: float,
                 alpha: float):
        """
        初始化
        :param in_features: 输入的特征数量
        :param out_features: 输出的特征数
        :param dropout: dropout
        :param alpha: leakyrelu 的 alpha 参数
        """
        super().__init__()

        self.dropout = Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.feed_forward = Linear(in_features, out_features, bias=False)

        self.a1 = Parameter(torch.Tensor(out_features, 1))
        self.a2 = Parameter(torch.Tensor(out_features, 1))

        self.leaky_relu = LeakyReLU(self.alpha)

        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化参数
        :return:
        """
        nn.init.xavier_uniform_(self.feed_forward.weight, gain=1.414)
        nn.init.xavier_uniform_(self.a1, gain=1.414)
        nn.init.xavier_uniform_(self.a2, gain=1.414)

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        模型运算
        :param input: (B, num_nodes, in_features)
        :param adj: 邻接矩阵 (B, num_nodes, num_nodes)
        :return: (B, node_len, out_features)
        """

        # 线性变换, 转换成隐层, (B, node_len, in_features) -> (B, node_len, out_features)
        h = self.feed_forward(input)

        # [batch_size, N, out_features]
        batch_size, num_nodes, _ = h.size()

        # 这里是将 a \cdot [Wf_i || W_fj] 这样的运算拆开来做了, 这就与 GAT 实际算法描述是一致的了
        #  a \cdot [Wf_i || W_fj] = a_1 * Wf_i + a_2 * Wf_j
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, num_nodes)
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, num_nodes).transpose(1, 2)

        # 计算 leaky_relu
        e = self.leaky_relu(middle_result1 + middle_result2)

        # 计算 attetion, 注意 为了使用 softmax, 所以这里将 邻接矩阵为 0 的，用极大的负值填充
        # 这样就会在计算 softmax 中为 0， 因为 exp(float("-inf"))=0
        attention1 = e.masked_fill(adj == 0, -1e9)

        # 计算 attention softmax
        attention = F.softmax(attention1, dim=2)
        attention = self.dropout(attention)

        h_prime = torch.matmul(attention, h)

        return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GAT(nn.Module):
    """
    图注意力模型，当前模型，最多支持两层
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 dropout: float,
                 alpha: float,
                 num_heads: int,
                 hidden_size: int = None):
        """
        初始化
        :param in_features: 输入的 node 维度
        :param out_features: 输出的 node 维度
        :param dropout: dropout
        :param alpha: 在 GraphAttentionLayer 中 LeakyRelu 用到的 alpha
        :param num_heads: 头的数量
        :param hidden_size: 隐层 size，如果是 None 表示没有隐层; 否则，只有一个隐层
        """

        super().__init__()
        self.dropout = Dropout(dropout)
        self.hidden_size = hidden_size
        self.activation = ELU()
        self.log_softmax = LogSoftmax(dim=2)
        if hidden_size is None:
            self.layers = ModuleList([GraphAttentionLayer(in_features=in_features,
                                                          out_features=out_features,
                                                          dropout=dropout,
                                                          alpha=alpha)
                                      for _ in range(num_heads)])
        else:
            self.layers = ModuleList([GraphAttentionLayer(in_features=in_features,
                                                          out_features=hidden_size,
                                                          dropout=dropout,
                                                          alpha=alpha)
                                      for _ in range(num_heads)])
            self.final_layer = GraphAttentionLayer(in_features=hidden_size * num_heads,
                                                   out_features=out_features,
                                                   dropout=dropout,
                                                   alpha=alpha)

    def forward(self, nodes: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        GAT 运算
        :param nodes: 图的节点，shape: (B, num_nodes, in_features)
        :param adj: 图的邻接矩阵，不包含 self loop
        :return: 计算后的结果, shape: (B, num_nodes, output_features)
        """

        assert nodes.dim() == 3, f"nodes 的维度: {nodes.dim()}, 与 (B, num_nodes, in_features) 不匹配"

        nodes = self.dropout(nodes)

        if self.hidden_size is None:
            nodes = torch.stack([self.activation(layer(nodes, adj)) for layer in self.layers], dim=2)
            nodes = nodes.sum(2)
            nodes = self.dropout(nodes)
        else:
            nodes = torch.cat([self.activation(layer(nodes, adj)) for layer in self.layers], dim=2)
            nodes = self.dropout(nodes)
            nodes = self.activation(self.final_layer(nodes, adj))
        return self.log_softmax(nodes)

