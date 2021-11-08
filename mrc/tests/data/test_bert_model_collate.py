#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
test bert model collate

Authors: PanXu
Date:    2021/10/25 17:24:00
"""
import os

from mrc.data.bert_model_collate import BertModelCollate
from easytext.utils.bert_tokenizer import bert_tokenizer
from easytext.utils.nn.tensor_util import is_tensor_equal

from mrc import ROOT_PATH

from mrc.tests import ASSERT


from mrc.tests.paper.collate_functions import collate_to_max_length


def test_bert_model_collate(mrc_msra_ner_dataset, paper_mrc_msra_ner_dataset):
    max_length = 128

    bert_dir = "data/pretrained/bert/chinese_roberta_wwm_large_ext_pytorch"
    bert_dir = os.path.join(ROOT_PATH, bert_dir)

    bert_model_collate = BertModelCollate(tokenizer=bert_tokenizer(bert_dir), max_length=max_length)

    instances = [instance for instance in mrc_msra_ner_dataset]
    model_inputs = bert_model_collate(instances=instances)

    inputs = model_inputs.model_inputs

    paper_instances = [instance for instance in paper_mrc_msra_ner_dataset]
    paper_model_inputs = collate_to_max_length(paper_instances)

    paper_token_ids = paper_model_inputs[0]
    token_ids = inputs["input_ids"]

    ASSERT.assertTrue(is_tensor_equal(paper_token_ids, token_ids, epsilon=0))

    paper_type_ids = paper_model_inputs[1]
    type_ids = inputs["token_type_ids"]

    ASSERT.assertTrue(is_tensor_equal(paper_type_ids, type_ids, epsilon=0))

    paper_start_label_indices = paper_model_inputs[2]

    start_label_indices = model_inputs.labels["start_position_labels"]

    ASSERT.assertTrue(is_tensor_equal(paper_start_label_indices, start_label_indices, epsilon=0))

    paper_end_label_indices = paper_model_inputs[3]

    end_label_indices = model_inputs.labels["end_position_labels"]

    ASSERT.assertTrue(is_tensor_equal(paper_end_label_indices, end_label_indices, epsilon=0))

    paper_start_label_mask = paper_model_inputs[4]
    sequence_mask = inputs["sequence_mask"].long()

    ASSERT.assertTrue(is_tensor_equal(paper_start_label_mask, sequence_mask, epsilon=0))

    paper_end_label_mask = paper_model_inputs[5]
    sequence_mask = inputs["sequence_mask"].long()

    ASSERT.assertTrue(is_tensor_equal(paper_end_label_mask, sequence_mask, epsilon=0))

    paper_match_labels = paper_model_inputs[6]
    match_labels = model_inputs.labels["match_position_labels"]

    ASSERT.assertTrue(is_tensor_equal(paper_match_labels, match_labels, epsilon=0))












