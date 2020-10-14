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


class Record:
    """
    训练中的记录数据, 用作 trainer callback 参数
    """

    def __init__(self,
                 epoch: int = None,
                 epoch_train_loss: float = None,
                 epoch_evaluate_loss: float = None
                 ):
        self.epoch = epoch
        self.epoch_train_loss = epoch_train_loss
        self.epoch_evaluate_loss = epoch_evaluate_loss

