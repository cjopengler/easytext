#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
数据配置

Authors: PanXu
Date:    2020/09/10 21:37:00
"""


class DatasetConfig:
    """
    dataset config
    """

    def __init__(self,
                 train_dataset_file_path: str,
                 validation_dataset_file_path: str):

        self.train_dataset_file_path = train_dataset_file_path
        self.validation_dataset_file_path = validation_dataset_file_path

