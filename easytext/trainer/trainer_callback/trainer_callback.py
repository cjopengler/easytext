#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
trainer 回调

Authors: PanXu
Date:    2020/10/13 14:36:00
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from easytext.trainer import Trainer

from easytext.trainer.record import Record
from easytext.distributed import Distributed


class TrainerCallback(Distributed):
    """
    trainer callbeck 接口
    """

    @property
    def is_distributed(self) -> bool:
        raise NotImplementedError()

    def on_train_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        """
        在每一个 epoch 训练开始时候会调用
        :param trainer: 训练器
        :param record: 训练中的记录数据
        :return:
        """
        raise NotImplemented()

    def on_train_epoch_stop(self, trainer: "Trainer", record: Record) -> None:
        """
        在每一个 epoch 训练结束时候会调用
        :param trainer: 训练器
        :param record: 训练中的记录数据
        :return:
        """
        raise NotImplemented()

    def on_evaluate_validation_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        """
        在每一个 epoch evaluate 验证集开始的时候会调用
        :param trainer: 训练器
        :param record: 训练中的记录数据
        :return:
        """
        raise NotImplemented()

    def on_evaluate_validation_epoch_stop(self, trainer: "Trainer", record: Record) -> None:
        """
        在每一个 epoch evaluate 验证集结束的时候会调用
        :param trainer: 训练器
        :param record: 训练中的记录数据
        :return:
        """
        raise NotImplemented()

    def on_training_complete(self, trainer: "Trainer", record: Record) -> None:
        """
        训练过程全部结束
        :param trainer: 训练器
        :param record: 训练中的记录数据
        :return:
        """
        raise NotImplemented

