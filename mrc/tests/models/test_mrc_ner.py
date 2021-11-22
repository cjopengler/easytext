#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
测试  mrc ner

Authors: PanXu
Date:    2021/11/07 11:45:00
"""

import torch

from mrc.models import MRCNer, MRCNerOutput


import os

import torch
import logging

from transformers import BertConfig

from easytext.utils.bert_tokenizer import bert_tokenizer
from easytext.utils.nn.tensor_util import is_tensor_equal

from mrc import ROOT_PATH

from mrc.tests import ASSERT
from mrc.data.bert_model_collate import BertModelCollate
from mrc.tests.paper.collate_functions import collate_to_max_length
from mrc.models import MRCNer, MRCNerOutput
from mrc.tests.paper.bert_query_ner import BertQueryNER
from mrc.tests.paper.query_ner_config import BertQueryNerConfig

from easytext.utils.seed_util import set_seed


def fake_model_weight(module: torch.nn.Module):

    if isinstance(module, torch.nn.Linear):
        fake_weight = torch.rand(module.weight.size())
        fake_bias = 0.
        module.weight.data.copy_(fake_weight)
        module.bias.data.fill_(fake_bias)


def test_mrc_ner(mrc_msra_ner_dataset, paper_mrc_msra_ner_dataset):

    # 设置 random seed 保证每一次的结果是一样的
    set_seed()

    max_length = 128

    bert_dir = "data/pretrained/bert/chinese_roberta_wwm_large_ext_pytorch"
    bert_dir = os.path.join(ROOT_PATH, bert_dir)

    bert_config = BertConfig.from_pretrained(bert_dir)

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

    logging.info(f"begin mrc ner")
    set_seed()

    mrc_model = MRCNer(bert_dir=bert_dir, dropout=0)

    # 设置固定权重
    set_seed()
    mrc_model.start_classifier.apply(fake_model_weight)
    mrc_model.end_classifier.apply(fake_model_weight)
    mrc_model.match_classifier.apply(fake_model_weight)

    # fake_start = torch.rand(mrc_model.start_classifier.weight.size())
    # mrc_model.start_classifier.weight.data.copy_(fake_start)

    # fake_end = torch.rand(mrc_model.end_classifier.weight.size())
    # mrc_model.end_classifier.weight.data.copy_(fake_end)

    # fake_match = torch.rand(mrc_model.match_classifier.weight)

    logging.info(f"mrc ner forward")
    mrc_model_output = mrc_model.forward(**model_inputs.model_inputs)

    logging.info(f"end mrc ner")

    logging.info(f"begin paper ner")

    set_seed()
    # 获取 bert config
    bert_config = BertQueryNerConfig.from_pretrained(bert_dir,
                                                     mrc_dropout=0)

    # 获取模型
    paper_model = BertQueryNER.from_pretrained(bert_dir, config=bert_config)

    # paper_model.start_outputs.weight.data.copy_(fake_start)
    set_seed()
    paper_model.start_outputs.apply(fake_model_weight)
    paper_model.end_outputs.apply(fake_model_weight)
    paper_model.span_embedding.apply(fake_model_weight)

    logging.info(f"paper ner forward")
    paper_attention_mask = (paper_token_ids != 0).long()
    paper_output = paper_model.forward(input_ids=paper_token_ids,
                                       token_type_ids=paper_type_ids,
                                       attention_mask=paper_attention_mask)

    paper_start_logits, paper_end_logits, paper_span_logits = paper_output

    logging.info(f"end paper ner")

    ASSERT.assertTrue(is_tensor_equal(mrc_model_output.start_logits, paper_start_logits, epsilon=1e-10))
    ASSERT.assertTrue(is_tensor_equal(mrc_model_output.end_logits, paper_end_logits, epsilon=1e-10))
    ASSERT.assertTrue(is_tensor_equal(mrc_model_output.match_logits, paper_span_logits, epsilon=1e-10))













