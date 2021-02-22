#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 model collate

Authors: PanXu
Date:    2020/07/13 20:19:00
"""

from torch.utils.data import DataLoader

from easytext.model import ModelInputs
from easytext.utils import log_util

from acsa.data import ACSAModelCollate

from acsa.tests import ASSERT

log_util.config()


def test_model_collate(acsa_sem_eval_dataset, vocabulary_builder):
    token_vocabulary = vocabulary_builder.token_vocabulary
    category_vocabulary = vocabulary_builder.category_vocabulary
    label_vocabulary = vocabulary_builder.label_vocabulary

    model_collate_fn = ACSAModelCollate(vocabulary_builder=vocabulary_builder)
    data_loader = DataLoader(dataset=acsa_sem_eval_dataset,
                             batch_size=10,
                             num_workers=0,
                             collate_fn=model_collate_fn)

    for model_inputs in data_loader:
        model_inputs: ModelInputs = model_inputs

        ASSERT.assertEqual(3, model_inputs.batch_size)

        expect_categories = ["service", "food", "anecdotes/miscellaneous"]
        categories = [category_vocabulary.token(category_index.item())
                      for category_index in model_inputs.model_inputs["category"]]
        ASSERT.assertListEqual(expect_categories, categories)

        expect_labels = ["negative", "positive", "negative"]
        labels = [label_vocabulary.token(label_index.item()) for label_index in model_inputs.labels]
        ASSERT.assertListEqual(expect_labels, labels)

        sentence_index_0 = model_inputs.model_inputs["sentence"][0]

        expect_sentence = ["but", "the", "staff", "was", "so", "horrible", "to", "us."]
        sentence_0 = [token_vocabulary.token(index.item()) for index in sentence_index_0[0:len(expect_sentence)]]

        ASSERT.assertListEqual(expect_sentence, sentence_0)
        ASSERT.assertEqual(0, sentence_index_0[len(expect_sentence):].sum())
