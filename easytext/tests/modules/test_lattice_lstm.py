#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
测试 lattice lstm

Authors: PanXu
Date:    2021/01/21 09:43:00
"""

import torch

from easytext.modules.lattice_lstm import WordLSTMCell

from easytext.tests import ASSERT


def test_word_lstm_cell_with_bias():
    """
    测试 WordLSTMCell
    :return:
    """

    input_size = 2
    hidden_size = 3
    word_lstm_cell = WordLSTMCell(input_size=input_size, hidden_size=hidden_size, bias=True)

    value = list()

    for i in range(input_size):
        value.append([j * 0.37 for j in range(i * hidden_size * 3, (i + 1) * hidden_size * 3)])

    with torch.no_grad():
        word_lstm_cell.weight_ih.copy_(torch.tensor(value, dtype=torch.float))

    word_input = torch.tensor([[0.2, 0.4]], dtype=torch.float)
    h = torch.tensor([[0.2, 0.11, 0.15]], dtype=torch.float)
    c = torch.tensor([[0.5, 0.6, 0.7]], dtype=torch.float)

    output_c = word_lstm_cell(input_=word_input,
                              hx=(h, c))

    expect_size = (1, hidden_size)
    ASSERT.assertEqual(expect_size, output_c.size())

    expect_output_c = [1.3054, 1.4113, 1.5386]

    for e_i, i in zip(expect_output_c, output_c[0].tolist()):
        ASSERT.assertAlmostEqual(e_i, i, places=3)


def test_word_lstm_cell_without_bias():
    """
    测试 WordLSTMCell
    :return:
    """

    hidden_size = 2
    word_lstm_cell = WordLSTMCell(input_size=2, hidden_size=hidden_size, bias=False)

    word_input = torch.tensor([[1., 2.]], dtype=torch.float)
    h = torch.tensor([[3., 4.]], dtype=torch.float)
    c = torch.tensor([[5, 6]], dtype=torch.float)

    output_c = word_lstm_cell(input_=word_input,
                              hx=(h, c))

    expect_size = (1, hidden_size)
    ASSERT.assertEqual(expect_size, output_c.size())



