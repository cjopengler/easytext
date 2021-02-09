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

from easytext.modules.lattice_lstm import WordLSTMCell, MultiInputLSTMCell

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
        torch.nn.init.constant(word_lstm_cell.bias, val=1.0)

    word_input = torch.tensor([[0.2, 0.4]], dtype=torch.float)
    h = torch.tensor([[0.2, 0.11, 0.15]], dtype=torch.float)
    c = torch.tensor([[0.5, 0.6, 0.7]], dtype=torch.float)

    output_c = word_lstm_cell(input_=word_input,
                              hx=(h, c))

    expect_size = (1, hidden_size)
    ASSERT.assertEqual(expect_size, output_c.size())

    expect_output_c = [1.4231, 1.5257, 1.6372]

    for e_i, i in zip(expect_output_c, output_c[0].tolist()):
        ASSERT.assertAlmostEqual(e_i, i, places=3)


def test_word_lstm_cell_without_bias():
    """
    测试 WordLSTMCell
    :return:
    """

    input_size = 2
    hidden_size = 3
    word_lstm_cell = WordLSTMCell(input_size=input_size, hidden_size=hidden_size, bias=False)

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


def test_multi_input_lstm_cell():
    """
    测试 MultiInputLSTMCell
    """

    input_size = 2
    hidden_size = 3

    cell = MultiInputLSTMCell(input_size=input_size, hidden_size=hidden_size, bias=True)

    with torch.no_grad():
        weight_ih_value = list()

        for i in range(input_size):
            weight_ih_value.append([j * 0.37 for j in range(i * hidden_size * 3, (i + 1) * hidden_size * 3)])

        cell.weight_ih.copy_(torch.tensor(weight_ih_value, dtype=torch.float))

        alpha_weight_ih_value = list()

        for i in range(input_size):
            alpha_weight_ih_value.append([j * 0.23 for j in range(i * hidden_size, (i + 1) * hidden_size)])

        cell.alpha_weight_ih.copy_(torch.tensor(alpha_weight_ih_value, dtype=torch.float))

        torch.nn.init.constant(cell.bias, val=1.0)
        torch.nn.init.constant(cell.alpha_bias, val=0.5)

    char_input = torch.tensor([[0.2, 0.4]], dtype=torch.float)

    h = torch.tensor([[0.2, 0.11, 0.15]], dtype=torch.float)
    c = torch.tensor([[0.5, 0.6, 0.7]], dtype=torch.float)

    word_c_input = [torch.tensor([[0.7, 0.5, 0.2]], dtype=torch.float),
                    torch.tensor([[0.3, 0.4, 1.5]], dtype=torch.float)]

    output_hc = cell(input_=char_input,
                     c_input=word_c_input,
                     hx=(h, c))

    expect_size = (1, hidden_size)

    ASSERT.assertEqual(expect_size, output_hc[0].size())
    ASSERT.assertEqual(expect_size, output_hc[1].size())

    expects = [[0.5728, 0.5523, 0.7130], [0.6873, 0.6506, 0.9345]]

    for expect, hc in zip(expects, output_hc):

        for e_i, hc_i in zip(expect, hc[0].tolist()):
            ASSERT.assertAlmostEqual(e_i, hc_i, places=4)


def test_multi_input_lstm_cell_without_bias():
    """
    测试 MultiInputLSTMCell
    """

    input_size = 2
    hidden_size = 3

    cell = MultiInputLSTMCell(input_size=input_size, hidden_size=hidden_size, bias=False)

    with torch.no_grad():
        weight_ih_value = list()

        for i in range(input_size):
            weight_ih_value.append([j * 0.37 for j in range(i * hidden_size * 3, (i + 1) * hidden_size * 3)])

        cell.weight_ih.copy_(torch.tensor(weight_ih_value, dtype=torch.float))

        alpha_weight_ih_value = list()

        for i in range(input_size):
            alpha_weight_ih_value.append([j * 0.23 for j in range(i * hidden_size, (i + 1) * hidden_size)])

        cell.alpha_weight_ih.copy_(torch.tensor(alpha_weight_ih_value, dtype=torch.float))

    char_input = torch.tensor([[0.2, 0.4]], dtype=torch.float)

    h = torch.tensor([[0.2, 0.11, 0.15]], dtype=torch.float)
    c = torch.tensor([[0.5, 0.6, 0.7]], dtype=torch.float)

    word_c_input = [torch.tensor([[0.7, 0.5, 0.2]], dtype=torch.float),
                    torch.tensor([[0.3, 0.4, 1.5]], dtype=torch.float)]

    output_hc = cell(input_=char_input,
                     c_input=word_c_input,
                     hx=(h, c))

    expect_size = (1, hidden_size)

    ASSERT.assertEqual(expect_size, output_hc[0].size())
    ASSERT.assertEqual(expect_size, output_hc[1].size())

    expects = [[0.5356, 0.5204, 0.6862], [0.6855, 0.6490, 0.9451]]

    for expect, hc in zip(expects, output_hc):

        for e_i, hc_i in zip(expect, hc[0].tolist()):
            ASSERT.assertAlmostEqual(e_i, hc_i, places=4)
