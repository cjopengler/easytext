#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
loss 曲线的 callback

Authors: PanXu
Date:    2020/10/16 17:55:00
"""

import logging
import time

from torch.utils.tensorboard import SummaryWriter
from torch import distributed as Distributed

from easytext.trainer import Record
from easytext.trainer.trainer_callback import TrainerCallback
from easytext.trainer.tensorboard_manager import MainTagManager


class DistributedCallback(TrainerCallback):
    """
    用于分布式训练中用到的 callback
    """

    def __init__(self, target_rank: int = 0):
        """
        初始化化
        :param target_rank: 目标 rank 进程中做 loss 相关操作
        """
        self._target_rank = target_rank

    def _is_target_rank(self) -> bool:
        """
        判断是否是目标 rank 进程
        :return: True: 是 target 进程; False: 不是 target 进程
        """
        return Distributed.get_rank() == self._target_rank

    def _on_train_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        raise NotImplementedError()

    def on_train_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        if self._is_target_rank():
            self._on_train_epoch_start(trainer=trainer, record=record)

    def _on_train_epoch_stop(self, trainer: "Trainer", record: Record) -> None:
        raise NotImplementedError()

    def on_train_epoch_stop(self, trainer: "Trainer", record: Record) -> None:
        if self._is_target_rank():
            return self._on_train_epoch_stop(trainer=trainer, record=record)

    def _on_evaluate_validation_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        raise NotImplementedError()

    def on_evaluate_validation_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        if self._is_target_rank():
            self._on_evaluate_validation_epoch_start(trainer=trainer, record=record)

    def _on_evaluate_validation_epoch_stop(self, trainer: "Trainer", record: Record) -> None:
        raise NotImplementedError()

    def on_evaluate_validation_epoch_stop(self, trainer: "Trainer", record: Record) -> None:
        if self._is_target_rank():
            self._on_evaluate_validation_epoch_stop(trainer=trainer, record=record)

    def _on_training_complete(self, trainer: "Trainer", record: Record) -> None:
        raise NotImplementedError()

    def on_training_complete(self, trainer: "Trainer", record: Record) -> None:
        if self._is_target_rank():
            self._on_training_complete(trainer=trainer, record=record)



