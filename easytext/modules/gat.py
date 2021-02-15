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
        :param input: (B, node_len, in_features)
        :param adj: 邻接矩阵 (B, node_num, node_num)
        :return: (B, node_len, out_features)
        """

        # 线性变换, 转换成隐层, (B, node_len, in_features) -> (B, node_len, out_features)
        h = self.feed_forward(input)

        # [batch_size, N, out_features]
        batch_size, node_num, _ = h.size()

        # 这里是将 a \cdot [Wf_i || W_fj] 这样的运算拆开来做了, 这就与 GAT 实际算法描述是一致的了
        #  a \cdot [Wf_i || W_fj] = a_1 * Wf_i + a_2 * Wf_j
        middle_result1 = torch.matmul(h, self.a1).expand(-1, -1, node_num)
        middle_result2 = torch.matmul(h, self.a2).expand(-1, -1, node_num).transpose(1, 2)

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
    图注意力模型，应用了多个 GraphAttentionLayer
    """

    def __init__(self, nfeat, nclass, dropout, alpha, nheads, layer, nhid=None):
        super().__init__()
        self.dropout = Dropout(dropout)
        self.layer = layer
        self.nhid = nhid
        if nhid is None:
            self.attentions = ModuleList([GraphAttentionLayer(nfeat, nclass, dropout=dropout, alpha=alpha) for _ in
                               range(nheads)])
        else:
            self.attentions = ModuleList([GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in
                               range(nheads)])
            self.out_att = GraphAttentionLayer(nhid * nheads, nclass, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)

        if self.nhid is None:
            x = torch.stack([F.elu(att(x, adj)) for att in self.attentions], dim=2)
            x = x.sum(2)
            x = F.dropout(x, self.dropout, training=self.training)
            return F.log_softmax(x, dim=2)
        else:
            x = torch.cat([F.elu(att(x, adj)) for att in self.attentions], dim=2)
            x = F.dropout(x, self.dropout, training=self.training)
            x = F.elu(self.out_att(x, adj))
            return F.log_softmax(x, dim=2)




