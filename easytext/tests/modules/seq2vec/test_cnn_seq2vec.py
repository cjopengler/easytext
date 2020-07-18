#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 cnn seq2vec

Authors: PanXu
Date:    2020/07/08 20:10:00
"""
import torch

from easytext.modules.seq2vec import CnnSeq2Vec
from easytext.tests import ASSERT


def test_cnn_seq2vec():
    """
    测试 cnn seq2vec
    :return:
    """

    encoder = CnnSeq2Vec(embedding_dim=2, num_filters=1, kernel_sizes=(1, 2))

    for name, parameter in encoder.named_parameters():
        parameter.data.fill_(1.)

    tokens = torch.FloatTensor([[[0.7, 0.8], [0.1, 1.5]]])
    vector = encoder(sequence=tokens, mask=None)
    vector = vector.view(-1).tolist()

    expect = torch.tensor([[0.1 + 1.5 + 1., 0.7 + 0.8 + 0.1 + 1.5 + 1.]]).view(-1).tolist()

    ASSERT.assertEqual(len(expect), len(vector))
    for i in range(len(vector)):
        ASSERT.assertAlmostEqual(expect[i], vector[i])


def test_cnn_seq2vec_with_mask():
    """
    测试带有 mask 的 cnn seq2vec
    :return:
    """

    encoder = CnnSeq2Vec(embedding_dim=2, num_filters=1, kernel_sizes=(1, 2))

    for name, parameter in encoder.named_parameters():
        parameter.data.fill_(1.)

    tokens = torch.FloatTensor([[[0.7, 0.8], [0.1, 1.5]]])
    mask = torch.ByteTensor([[1, 0]])
    vector = encoder(sequence=tokens, mask=mask)
    vector = vector.view(-1).tolist()

    expect = torch.tensor([[0.7 + 0.8 + 1., 0.7 + 0.8 + 0. + 0. + 1.]]).view(-1).tolist()

    ASSERT.assertEqual(len(expect), len(vector))
    for i in range(len(vector)):
        ASSERT.assertAlmostEqual(expect[i], vector[i])


def test_cnn_seq2vec_output_dim():
    """
    测试 cnn 输出维度
    :return:
    """
    kernel_size = (1, 2, 3, 4, 5)
    encoder = CnnSeq2Vec(embedding_dim=7,
                         num_filters=13,
                         kernel_sizes=kernel_size)

    tokens = torch.rand(4, 8, 7)
    vector = encoder(sequence=tokens, mask=None)
    expect = (4, 13 * len(kernel_size))

    ASSERT.assertEqual(expect, vector.size())
