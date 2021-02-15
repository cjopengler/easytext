#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
测试 gat

Authors: PanXu
Date:    2021/02/15 11:19:00
"""

import torch

from easytext.modules import GraphAttentionLayer
from easytext.utils.nn import tensor_util
from easytext.tests import ASSERT


def test_graph_attention_layer():
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)

    in_features = 2
    out_features = 4

    gat_layer = GraphAttentionLayer(in_features=in_features, out_features=out_features, dropout=0.0, alpha=0.1)

    nodes = torch.tensor([[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                          [[0.7, 0.8], [0.9, 0.10], [0.11, 0.12]]],
                         dtype=torch.float)

    adj = torch.tensor([[[0, 1, 0],
                         [1, 0, 0],
                         [0, 0, 0]],
                        [[0, 1, 1],
                         [1, 0, 1],
                         [1, 1, 0]]],
                       dtype=torch.long)

    outputs: torch.Tensor = gat_layer(input=nodes, adj=adj)

    expect_size = (nodes.size(0), nodes.size(1), out_features)

    ASSERT.assertEqual(expect_size, outputs.size())

    # 下面的 expect 是从原论文中测试得到的结果，直接拿来用
    expect = torch.tensor([[[0.2831, 0.3588, -0.5131, -0.2058],
                            [0.1606, 0.1292, -0.2264, -0.0951],
                            [0.2831, 0.3588, -0.5131, -0.2058]],

                           [[-0.0748, 0.5025, -0.3840, -0.1192],
                            [0.2959, 0.4624, -0.6123, -0.2405],
                            [0.1505, 0.8668, -0.8609, -0.3059]]], dtype=torch.float)

    ASSERT.assertTrue(tensor_util.is_tensor_equal(expect, outputs, epsilon=1e-4))
