#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 sequence max label index decoder

Authors: PanXu
Date:    2020/07/05 16:42:00
"""

import torch
from easytext.tests import ASSERT

from easytext.label_decoder import SequenceMaxLabelIndexDecoder
from easytext.data import LabelVocabulary


def test_sequence_max_label_index_decoder():
    label_vocabulary = LabelVocabulary([["B-T", "B-T", "B-T", "I-T", "I-T", "O"]],
                                       padding=LabelVocabulary.PADDING)

    b_index = label_vocabulary.index("B-T")
    ASSERT.assertEqual(0, b_index)
    i_index = label_vocabulary.index("I-T")
    ASSERT.assertEqual(1, i_index)
    o_index = label_vocabulary.index("O")
    ASSERT.assertEqual(2, o_index)

    # [[O, B, I], [B, B, I], [B, I, I], [B, I, O]]
    batch_sequence_logits = torch.tensor([[[0.2, 0.3, 0.4], [0.7, 0.2, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.7, 0.2, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.1, 0.7, 0.3], [0.2, 0.3, 0.1]],
                                          [[0.8, 0.3, 0.4], [0.1, 0.7, 0.3], [0.2, 0.3, 0.5]]],
                                         dtype=torch.float)

    expect_sequence_labels = [["O", "B-T", "I-T"],
                              ["B-T", "B-T", "I-T"],
                              ["B-T", "I-T", "I-T"],
                              ["B-T", "I-T", "O"]]

    expect = list()

    for expect_sequence_label in expect_sequence_labels:
        expect.append([label_vocabulary.index(label) for label in expect_sequence_label])

    decoder = SequenceMaxLabelIndexDecoder(label_vocabulary=label_vocabulary)

    label_indices = decoder(logits=batch_sequence_logits,
                            mask=None)

    ASSERT.assertEqual(expect, label_indices.tolist())
