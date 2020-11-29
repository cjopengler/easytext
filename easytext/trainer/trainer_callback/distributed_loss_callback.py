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

import torch
from torch.utils.tensorboard import SummaryWriter
from torch import distributed as Distributed

from easytext.trainer import Record
from easytext.trainer.trainer_callback import DistributedCallback
from easytext.trainer.trainer_callback import LossCallback
from easytext.trainer.tensorboard_manager import MainTagManager


class DistributedLossCallback(DistributedCallback):
    """
    用于分布式训练, loss callback 用来统计 epoch loss 变化
    """

    def __init__(self, tensorboard_summary_writer: SummaryWriter, target_rank: int = 0):
        """
        初始化化
        :param target_rank: 目标 rank 进程中做 loss 相关操作
        :param tensorboard_summary_writer: 对于非 target_rank 进程，该参数应该是 None, 会做校验
        """
        super().__init__(target_rank=target_rank)

        if self._is_target_rank():
            assert tensorboard_summary_writer is not None, f"在 {self._target_rank} 进程中, SummaryWriter 不能为 None."
        else:
            assert tensorboard_summary_writer is None, f"在 {self._target_rank} 进程中, SummaryWriter 必须设置为 None."

        self._loss_callback = LossCallback(tensorboard_summary_writer=tensorboard_summary_writer)

    def _on_train_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        self._loss_callback.on_train_epoch_start(trainer=trainer, record=record)

    def _on_train_epoch_stop(self, trainer: "Trainer", record: Record) -> None:
        self._loss_callback.on_train_epoch_stop(trainer=trainer, record=record)

    def on_train_epoch_stop(self, trainer: "Trainer", record: Record) -> None:

        loss = torch.tensor(record.epoch_train_loss, dtype=torch.float)

        if Distributed.get_backend() == "nccl":
            loss.to(Distributed.get_rank())

        Distributed.reduce(tensor=loss, dst=self._target_rank)

    def _on_evaluate_validation_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        self._loss_callback.on_evaluate_validation_epoch_start(trainer=trainer, record=record)

    def _on_evaluate_validation_epoch_stop(self, trainer: "Trainer", record: Record) -> None:
        self._loss_callback.on_evaluate_validation_epoch_stop(trainer=trainer, record=record)

    def _on_training_complete(self, trainer: "Trainer", record: Record) -> None:
        self._loss_callback.on_training_complete(trainer=trainer, record=record)



