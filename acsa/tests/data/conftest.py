#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
conf test

Authors: PanXu
Date:    2020/07/12 15:57:00
"""
import os
import pytest

from torch.utils.data import DataLoader

from easytext.data import Vocabulary
from easytext.data import LabelVocabulary

from acsa import ROOT_PATH
from acsa.data.dataset import SemEvalDataset
from acsa.data.dataset import ACSASemEvalDataset

from acsa.data import VocabularyCollate


@pytest.fixture(scope="package")
def sem_eval_dataset():
    sample_dataset_file_path = "data/dataset/SemEval-2014-Task-4-REST/sample.xml"
    sample_dataset_file_path = os.path.join(ROOT_PATH, sample_dataset_file_path)

    dataset = SemEvalDataset(dataset_file_path=sample_dataset_file_path)

    return dataset


@pytest.fixture(scope="package")
def acsa_sem_eval_dataset():
    sample_dataset_file_path = "data/dataset/SemEval-2014-Task-4-REST/sample.xml"
    sample_dataset_file_path = os.path.join(ROOT_PATH, sample_dataset_file_path)

    dataset = ACSASemEvalDataset(dataset_file_path=sample_dataset_file_path)

    return dataset


@pytest.fixture(scope="package")
def vocabulary(acsa_sem_eval_dataset):
    data_loader = DataLoader(acsa_sem_eval_dataset,
                             batch_size=10,
                             num_workers=0,
                             collate_fn=VocabularyCollate())

    batch_tokens = list()
    batch_categories = list()
    batch_labels = list()

    for vocab_dict in data_loader:

        tokens = vocab_dict["tokens"]
        batch_tokens.append(tokens)

        categories = vocab_dict["categories"]
        batch_categories.append(categories)

        labels = vocab_dict["labels"]
        batch_labels.append(labels)

    token_vocabulary = Vocabulary(tokens=batch_tokens,
                                  padding=Vocabulary.PADDING,
                                  unk=Vocabulary.UNK,
                                  special_first=True)
    category_vocabulary = LabelVocabulary(labels=batch_categories, padding=None)
    label_vocabulary = LabelVocabulary(labels=batch_labels, padding=None)

    return {
        "token_vocabulary": token_vocabulary,
        "category_vocabulary": category_vocabulary,
        "label_vocabulary": label_vocabulary
    }
