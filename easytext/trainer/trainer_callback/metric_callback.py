#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
metric callback

Authors: PanXu
Date:    2020/10/19 19:55:00
"""

import logging
import time

from torch.utils.tensorboard import SummaryWriter

from easytext.trainer import Record
from easytext.trainer.trainer_callback import TrainerCallback
from easytext.trainer.tensorboard_manager import MainTagManager


class MetricCallback(TrainerCallback):
    """
    绘制 train 和 evaluate metric 曲线
    """

    def __init__(self, tensorboard_summary_writer: SummaryWriter):
        self._writer = tensorboard_summary_writer

    def on_train_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        pass

    def on_train_epoch_stop(self, trainer: "Trainer", record: Record) -> None:

        self._writer.add_scalars(main_tag=MainTagManager.EPOCH_METRIC,
                                 tag_scalar_dict={"train": record.train_target_metric.value},
                                 global_step=record.epoch)

    def on_evaluate_validation_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        pass

    def on_evaluate_validation_epoch_stop(self, trainer: "Trainer", record: Record) -> None:

        self._writer.add_scalars(main_tag=MainTagManager.EPOCH_METRIC,
                                 tag_scalar_dict={"validation": record.validation_target_metric.value},
                                 global_step=record.epoch)

    def on_training_complete(self, trainer: "Trainer", record: Record) -> None:
        # 注意这里要 flush 不然可能会导致没有输出
        self._writer.flush()

