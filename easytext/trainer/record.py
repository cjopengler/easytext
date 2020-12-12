#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
trainer 中在训练过程中的 记录

Authors: PanXu
Date:    2020/10/14 10:58:00
"""

from typing import Dict

from easytext.metrics import ModelTargetMetric


class Record:
    """
    训练中的记录数据, 用作 trainer callback 参数
    """

    def __init__(self):
        self.epoch: int = None
        self.epoch_train_num: int = None
        self.epoch_train_loss: float = None
        self.epoch_validation_num: int = None
        self.epoch_validation_loss: float = None

        self.train_metric: Dict = None
        self.train_target_metric: ModelTargetMetric = None
        self.validation_metric: Dict = None
        self.validation_target_metric: ModelTargetMetric = None

