#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
brief

Authors: PanXu
Date:    2020/06/27 00:12:00
"""

import os
from typing import Dict, Union

import pytest

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tokenizers import BertWordPieceTokenizer

from easytext.data import Vocabulary, LabelVocabulary, PretrainedVocabulary

from ner.data import VocabularyCollate

from easytext.data import Vocabulary

from mrc import ROOT_PATH
from mrc.data import MSRAFlatNerDataset


from mrc.tests.data.paper_src.mrc_ner_dataset import MRCNERDataset


@pytest.fixture(scope="session")
def mrc_msra_ner_dataset() -> MSRAFlatNerDataset:
    """
    数据集生成
    :return: msra flat ner dataset
    """
    dataset_file_path = "data/dataset/mrc_msra_ner/sample.json"
    dataset_file_path = os.path.join(ROOT_PATH, dataset_file_path)

    return MSRAFlatNerDataset(is_training=True, dataset_file_path=dataset_file_path)


@pytest.fixture(scope="session")
def bert_tokenizer():
    """
    bert tokenizer
    :return: bert tokenizer
    """
    bert_dir = os.path.join(ROOT_PATH,
                            "data/pretrained/bert/chinese_roberta_wwm_large_ext_pytorch")
    return BertTokenizer.from_pretrained(bert_dir)


@pytest.fixture(scope="session")
def paper_mrc_msra_ner_dataset() -> MRCNERDataset:
    """
    数据集生成
    :return: msra flat ner dataset
    """
    dataset_file_path = "data/dataset/mrc_msra_ner/sample.json"
    dataset_file_path = os.path.join(ROOT_PATH, dataset_file_path)

    vocab_path = os.path.join(ROOT_PATH,
                            "data/pretrained/bert/chinese_roberta_wwm_large_ext_pytorch/vocab.txt")

    tokenizer = BertWordPieceTokenizer(vocab_path)
    return MRCNERDataset(json_path=dataset_file_path, tokenizer=tokenizer, is_chinese=True)


