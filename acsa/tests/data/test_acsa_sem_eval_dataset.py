#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 acsa sem eval dataset

Authors: PanXu
Date:    2020/07/12 16:35:00
"""

from acsa.tests import ASSERT


def test_acsa_sem_eval_dataset(acsa_sem_eval_dataset):
    """
    测试 acsa sem eval dataset
    :param acsa_sem_eval_dataset:
    :return:
    """

    ASSERT.assertEqual(3, len(acsa_sem_eval_dataset))

    instance1 = acsa_sem_eval_dataset[1]

    expect_sentence = "To be completely fair, the only redeeming factor was the food, which was above average, " \
                      "but couldn't make up for all the other deficiencies of Teodora."
    ASSERT.assertEqual(instance1["sentence"], expect_sentence)
    ASSERT.assertEqual(instance1["category"], "food")
    ASSERT.assertEqual(instance1["label"], "positive")

    instance2 = acsa_sem_eval_dataset[2]
    ASSERT.assertEqual(instance2["sentence"], expect_sentence)
    ASSERT.assertEqual(instance2["category"], "anecdotes/miscellaneous")
    ASSERT.assertEqual(instance2["label"], "negative")
