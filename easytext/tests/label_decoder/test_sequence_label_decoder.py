#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 sequence label decoder

Authors: PanXu
Date:    2020/07/05 17:09:00
"""
import torch

from easytext.tests import ASSERT

from easytext.data import LabelVocabulary
from easytext.label_decoder import SequenceLabelDecoder


def test_sequence_label_decoder():
    """
    测试 sequence label decoder
    :return:
    """
    sequence_label_list = list()
    expect_spans = list()

    sequence_label = ["B-T", "I-T", "O-T"]
    expect = [{"label": "T", "begin": 0, "end": 2}]

    sequence_label_list.append(sequence_label)
    expect_spans.append(expect)

    sequence_label = ["B-T", "I-T", "I-T"]
    expect = [{"label": "T", "begin": 0, "end": 3}]

    sequence_label_list.append(sequence_label)
    expect_spans.append(expect)

    sequence_label = ["B-T", "I-T", "I-T", "B-T"]
    expect = [{"label": "T", "begin": 0, "end": 3},
              {"label": "T", "begin": 3, "end": 4}]

    sequence_label_list.append(sequence_label)
    expect_spans.append(expect)

    label_vocabulary = LabelVocabulary(sequence_label_list,
                                       padding=LabelVocabulary.PADDING)

    sequence_label_indices = list()
    mask_list = list()

    max_sequence_len = 4
    for sequence_labels in sequence_label_list:
        sequence_label_index = [label_vocabulary.index(label) for label in sequence_labels]

        mask = [1] * len(sequence_label_index) + [0] * (max_sequence_len - len(sequence_label_index))

        sequence_label_index.extend([label_vocabulary.padding_index] * (max_sequence_len - len(sequence_label_index)))

        sequence_label_indices.append(sequence_label_index)
        mask_list.append(mask)

    sequence_label_indices = torch.tensor(sequence_label_indices, dtype=torch.long)
    mask = torch.tensor(mask_list, dtype=torch.uint8)

    decoder = SequenceLabelDecoder(label_vocabulary=label_vocabulary)

    spans = decoder(label_indices=sequence_label_indices, mask=mask)

    ASSERT.assertListEqual(expect_spans, spans)
