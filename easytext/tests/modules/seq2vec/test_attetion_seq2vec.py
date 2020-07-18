#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 attention seq2vec

Authors: PanXu
Date:    2020/07/18 16:06:00
"""
import pytest

import torch
from torch.nn import Parameter
from torch import FloatTensor, LongTensor, ByteTensor

from easytext.modules.seq2vec import AttentionSeq2Vec

from easytext.tests import ASSERT


@pytest.fixture(scope="package")
def inputs():
    sequence = torch.FloatTensor(
        [
            [[1., 2.], [3., 4.], [5., 6.], [9., 8.]],
            [[7., 1.], [2., 8.], [9., 3.], [5., 2.]]
        ]
    )
    mask = torch.ByteTensor(
        [
            [1, 1, 1, 0],
            [1, 1, 0, 0]
        ]
    )
    return sequence, mask


def test_attention_seq2vec_with_mask(inputs):
    """
    测试 attention seq2vec
    :return:
    """

    sequence, mask = inputs

    encoder = AttentionSeq2Vec(input_size=2,
                               query_hidden_size=3,
                               value_hidden_size=None)

    encoder.wk.weight = Parameter(FloatTensor([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]
    ]))
    encoder.wk.bias = Parameter(FloatTensor([0.2, 0.4, 0.6]))

    encoder.attention.weight = Parameter(FloatTensor(
        [
            [0.6, 0.2, 7]
        ]
    ))

    vec = encoder(sequence=sequence, mask=mask)

    ASSERT.assertEqual((2, 2), vec.size())

    expect = torch.tensor([[3.23450, 4.23450],
                           [4.37470, 4.67550]])

    vec1d = vec.view(-1).tolist()
    expect1d = expect.view(-1).tolist()

    for expect_data, vec_data in zip(expect1d, vec1d):
        ASSERT.assertAlmostEqual(expect_data, vec_data, delta=1e-4)


def test_attention_seq2vec_no_mask(inputs):
    """
    测试 attention seq2vec
    :return:
    """

    sequence, mask = inputs

    encoder = AttentionSeq2Vec(input_size=2,
                               query_hidden_size=3,
                               value_hidden_size=None)

    encoder.wk.weight = Parameter(FloatTensor([
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6]
    ]))
    encoder.wk.bias = Parameter(FloatTensor([0.2, 0.4, 0.6]))

    encoder.attention.weight = Parameter(FloatTensor(
        [
            [0.6, 0.2, 7]
        ]
    ))

    vec = encoder(sequence=sequence, mask=None)

    print(vec)

    ASSERT.assertEqual((2, 2), vec.size())

    expect = torch.tensor([[4.8455, 5.2867],
                           [5.7232, 3.6037]])

    vec1d = vec.view(-1).tolist()
    expect1d = expect.view(-1).tolist()

    for expect_data, vec_data in zip(expect1d, vec1d):
        ASSERT.assertAlmostEqual(expect_data, vec_data, delta=1e-4)
