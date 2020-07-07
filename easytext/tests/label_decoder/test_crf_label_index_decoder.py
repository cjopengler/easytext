#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 label index decoder

Authors: PanXu
Date:    2020/07/05 15:10:00
"""
import pytest
import torch

from easytext.tests import ASSERT

from easytext.data import LabelVocabulary
from easytext.modules import ConditionalRandomField
from easytext.label_decoder import CRFLabelIndexDecoder


class CRFData:
    """
    测试用的 crf 数据
    """

    def __init__(self):
        bio_labels = [["O", "I-X", "B-X", "I-Y", "B-Y"]]

        self.label_vocabulary = LabelVocabulary(labels=bio_labels,
                                                padding=LabelVocabulary.PADDING)

        self.logits = torch.tensor([
            [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
            [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
        ], dtype=torch.float)

        self.tags = torch.tensor([
            [2, 3, 4],
            [3, 2, 2]
        ], dtype=torch.long)

        self.transitions = torch.tensor([
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.8, 0.3, 0.1, 0.7, 0.9],
            [-0.3, 2.1, -5.6, 3.4, 4.0],
            [0.2, 0.4, 0.6, -0.3, -0.4],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ], dtype=torch.float)

        self.transitions_from_start = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.6], dtype=torch.float)
        self.transitions_to_end = torch.tensor([-0.1, -0.2, 0.3, -0.4, -0.4], dtype=torch.float)

        # Use the CRF Module with fixed transitions to compute the log_likelihood
        self.crf = ConditionalRandomField(5)
        self.crf.transitions = torch.nn.Parameter(self.transitions)
        self.crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        self.crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)

        # constraint crf
        constraints = {(0, 0), (0, 1),
                       (1, 1), (1, 2),
                       (2, 2), (2, 3),
                       (3, 3), (3, 4),
                       (4, 4), (4, 0)}

        # Add the transitions to the end tag
        # and from the start tag.
        for i in range(5):
            constraints.add((5, i))
            constraints.add((i, 6))

        constraint_crf = ConditionalRandomField(num_tags=5, constraints=constraints)
        constraint_crf.transitions = torch.nn.Parameter(self.transitions)
        constraint_crf.start_transitions = torch.nn.Parameter(self.transitions_from_start)
        constraint_crf.end_transitions = torch.nn.Parameter(self.transitions_to_end)
        self.constraint_crf = constraint_crf


@pytest.fixture(scope="class")
def crf_data():
    """
    产生测试用的 crf data
    :return:
    """
    return CRFData()


def test_crf_label_index_decoder(crf_data):
    """
    测试 crf label index decoder
    :param crf_data: crf data
    :return:
    """
    mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 0]
    ], dtype=torch.long)

    crf_label_index_decoder = CRFLabelIndexDecoder(crf=crf_data.crf,
                                                   label_vocabulary=crf_data.label_vocabulary)

    label_indices = crf_label_index_decoder(logits=crf_data.logits,
                                            mask=mask)
    padding_index = crf_data.label_vocabulary.padding_index
    expect = [[2, 4, 3], [4, 2, padding_index]]

    ASSERT.assertListEqual(expect, label_indices.tolist())


def test_crf_label_index_decoder_with_constraint(crf_data):
    mask = torch.tensor([
        [1, 1, 1],
        [1, 1, 0]
    ], dtype=torch.uint8)

    crf_label_index_decoder = CRFLabelIndexDecoder(crf=crf_data.constraint_crf,
                                                   label_vocabulary=crf_data.label_vocabulary)

    label_indices = crf_label_index_decoder(logits=crf_data.logits,
                                            mask=mask)
    padding_index = crf_data.label_vocabulary.padding_index
    expect = [[2, 3, 3], [2, 3, padding_index]]

    ASSERT.assertListEqual(expect, label_indices.tolist())
