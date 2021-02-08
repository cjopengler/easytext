#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
conf test

Authors: PanXu
Date:    2021/02/07 09:13:00
"""
import os
import pytest

from easytext.data.pretrained_word_embedding_loader import GeneralPretrainedWordEmbeddingLoader

from ner import ROOT_PATH
from ner.data.dataset import LatticeNerDemoDataset


@pytest.fixture(scope="session")
def lattice_ner_demo_dataset() -> LatticeNerDemoDataset:
    """
    数据集生成
    :return: conll2003 数据集
    """
    dataset_file_path = "data/dataset/lattice_ner/demo.train.char"
    dataset_file_path = os.path.join(ROOT_PATH, dataset_file_path)

    return LatticeNerDemoDataset(dataset_file_path=dataset_file_path)


@pytest.fixture(scope="session")
def gaz_pretrained_embedding_loader() -> GeneralPretrainedWordEmbeddingLoader:
    pretrained_file_path = "data/ner/lattice_ner/ctb.50d.vec"
    pretrained_file_path = os.path.join(ROOT_PATH, pretrained_file_path)

    pretrained_embedding_loader = GeneralPretrainedWordEmbeddingLoader(embedding_dim=50,
                                                                       pretrained_file_path=pretrained_file_path)

    pretrained_embedding_loader.load()
    return pretrained_embedding_loader

