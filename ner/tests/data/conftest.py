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

from easytext.data import Vocabulary, LabelVocabulary, PretrainedVocabulary

from ner.data import VocabularyCollate

from easytext.data import Vocabulary

from ner import ROOT_PATH
from ner.data.dataset import Conll2003Dataset


@pytest.fixture(scope="session")
def conll2003_dataset() -> Conll2003Dataset:
    """
    数据集生成
    :return: conll2003 数据集
    """
    dataset_file_path = "data/conll2003/sample.txt"
    dataset_file_path = os.path.join(ROOT_PATH, dataset_file_path)

    return Conll2003Dataset(dataset_file_path=dataset_file_path)


@pytest.fixture(scope="session")
def vocabulary(conll2003_dataset) -> Dict[str, Union[Vocabulary, PretrainedVocabulary]]:
    data_loader = DataLoader(dataset=conll2003_dataset,
                             batch_size=2,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=VocabularyCollate())

    batch_tokens = list()
    batch_sequence_labels = list()

    for collate_dict in data_loader:
        batch_tokens.extend(collate_dict["tokens"])
        batch_sequence_labels.extend(collate_dict["sequence_labels"])

    token_vocabulary = Vocabulary(tokens=batch_tokens,
                                  padding=Vocabulary.PADDING,
                                  unk=Vocabulary.UNK,
                                  special_first=True)

    label_vocabulary = LabelVocabulary(labels=batch_sequence_labels,
                                       padding=LabelVocabulary.PADDING)
    return {"token_vocabulary": token_vocabulary,
            "label_vocabulary": label_vocabulary}
