#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
测试 gaz vocabulary collate

Authors: PanXu
Date:    2021/02/02 10:11:00
"""

from ner.data.lattice import GazVocabularyCollate

from ner.tests.data.conftest import msra_dataset


def test_gaz_vocabulary_collate(msra_dataset):

    gaz_vocabulary_collate = GazVocabularyCollate()

    result = gaz_vocabulary_collate(msra_dataset)
