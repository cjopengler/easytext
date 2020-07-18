#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 sem eval dataset

Authors: PanXu
Date:    2020/07/12 15:55:00
"""
from acsa.tests import ASSERT


def test_sem_eval_dataset(sem_eval_dataset):
    ASSERT.assertEqual(2, len(sem_eval_dataset))

    instance1 = sem_eval_dataset[1]

    expect = {
        "sentence": "To be completely fair, the only redeeming factor was the food, which was above average, "
                    "but couldn't make up for all the other deficiencies of Teodora.",

        "aspect_terms": [
            {"term": "food",
             "begin": 57,
             "end": 61,
             "polarity": "positive"}
        ],

        "aspect_categories": [
            {
                "category": "food",
                "polarity": "positive"
            },
            {
                "category": "anecdotes/miscellaneous",
                "polarity": "negative"
            }
        ]}

    ASSERT.assertDictEqual(expect, instance1)

