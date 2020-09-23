#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
conll2003 dataset config

Authors: PanXu
Date:    2020/09/10 21:41:00
"""
import os

from ner import ROOT_PATH
from ner.config.dataset_config import DatasetConfig


class Conll2003DatasetConfig(DatasetConfig):
    """
    conll2003 dataset config
    """

    def __init__(self, debug: bool):
        self.debug = debug

        if self.debug:
            train_dataset_file_path = "data/dataset/conll2003/sample.txt"
        else:
            train_dataset_file_path = "data/dataset/conll2003/eng.train"

        train_dataset_file_path = os.path.join(ROOT_PATH, train_dataset_file_path)

        if self.debug:
            validation_dataset_file_path = "data/dataset/conll2003/sample.txt"
        else:
            validation_dataset_file_path = "data/dataset/conll2003/eng.testa"
        validation_dataset_file_path = os.path.join(ROOT_PATH, validation_dataset_file_path)

        super().__init__(train_dataset_file_path=train_dataset_file_path,
                         validation_dataset_file_path=validation_dataset_file_path)




