#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 dynamic rnn

Authors: PanXu
Date:    2020/10/20 17:26:00
"""
import logging
import pytest

import torch
from torch.nn import LSTM, GRU, RNN

from easytext.modules import DynamicRnn, DynamicRnnOutput
from easytext.utils.json_util import json2str

from easytext.tests import ASSERT


@pytest.fixture(scope="package")
def sequence_embedding():
    sequence = torch.tensor([
        [
            [1, 2], [3, 4], [0, 0]
        ],

        [
            [9, 10], [0, 0], [0, 0]
        ],
        [
            [5, 6], [7, 8], [0, 0]
        ]], dtype=torch.float)

    mask = torch.tensor([
        [True, True, False],
        [True, False, False],
        [True, True, False]
    ], dtype=torch.bool)

    return sequence, mask


def test_dynamic_lstm(sequence_embedding):
    sequence, mask = sequence_embedding

    hidden_size = 4
    batch_size = 3
    sequence_len = 3

    lstm = LSTM(input_size=2,
                hidden_size=4,
                num_layers=2,
                batch_first=True,
                bidirectional=True)

    dynamic_rnn = DynamicRnn(rnn=lstm)

    rnn_output: DynamicRnnOutput = dynamic_rnn(sequence=sequence, mask=mask)

    logging.info(json2str(rnn_output))

    last_layer_h_n: torch.Tensor = rnn_output.last_layer_h_n

    last_layer_h_n_expect_size = (batch_size, hidden_size * 2)

    ASSERT.assertEqual(last_layer_h_n_expect_size, last_layer_h_n.size())

    last_layer_c_n: torch.Tensor = rnn_output.last_layer_c_n
    ASSERT.assertEqual(last_layer_h_n_expect_size, last_layer_c_n.size())

    sequence_encoding_expect_size = (batch_size, sequence_len, hidden_size * 2)
    senquence_encoding = rnn_output.output
    ASSERT.assertEqual(sequence_encoding_expect_size, senquence_encoding.size())


def test_dynamic_rnn(sequence_embedding):
    sequence, mask = sequence_embedding

    hidden_size = 4
    batch_size = 3
    sequence_len = 3

    rnn = RNN(input_size=2,
              hidden_size=4,
              num_layers=2,
              batch_first=True,
              bidirectional=True)

    dynamic_rnn = DynamicRnn(rnn=rnn)

    rnn_output: DynamicRnnOutput = dynamic_rnn(sequence=sequence, mask=mask)

    logging.info(json2str(rnn_output))

    last_layer_h_n: torch.Tensor = rnn_output.last_layer_h_n

    last_layer_h_n_expect_size = (batch_size, hidden_size * 2)

    ASSERT.assertEqual(last_layer_h_n_expect_size, last_layer_h_n.size())

    ASSERT.assertTrue(rnn_output.last_layer_c_n is None)

    sequence_encoding_expect_size = (batch_size, sequence_len, hidden_size * 2)
    senquence_encoding = rnn_output.output
    ASSERT.assertEqual(sequence_encoding_expect_size, senquence_encoding.size())
