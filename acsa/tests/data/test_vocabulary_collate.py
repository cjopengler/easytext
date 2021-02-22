#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 vocabulary collate

Authors: PanXu
Date:    2020/07/13 10:22:00
"""
from torch.utils.data import DataLoader

from acsa.data import VocabularyCollate

from acsa.tests import ASSERT


def test_vocabulary_collate(acsa_sem_eval_dataset):
    """
    测试 vocabulary collate
    :param acsa_sem_eval_dataset:
    :return:
    """

    collate_fn = VocabularyCollate()

    data_loader = DataLoader(dataset=acsa_sem_eval_dataset,
                             batch_size=10,
                             num_workers=0,
                             collate_fn=collate_fn)

    for vocab_dict in data_loader:

        tokens = vocab_dict["tokens"]
        ASSERT.assertTrue(len(tokens) > 10)

        categories = vocab_dict["categories"]
        expect_categories = ["service", "food", "anecdotes/miscellaneous"]
        ASSERT.assertListEqual(expect_categories, categories)

        labels = vocab_dict["labels"]
        expect_labels = ["negative", "positive", "negative"]
        ASSERT.assertListEqual(expect_labels, labels)


