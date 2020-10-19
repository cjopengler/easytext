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

from easytext.trainer import Record
from easytext.trainer.trainer_callback import TrainerCallback
from easytext.trainer.tensorboard_manager import MainTagManager


class LossCallback(TrainerCallback):
    """
    时间消耗 callback 用来统计不同阶段的时间, 时间单位是 毫秒 ms
    """

    def __init__(self, tensorboard_summary_writer: SummaryWriter):
        self._writer = tensorboard_summary_writer

    def on_train_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        pass

    def on_train_epoch_stop(self, trainer: "Trainer", record: Record) -> None:

        logging.info(f"Epoch {record.epoch} train loss: {record.epoch_train_loss:.4f}")

        self._writer.add_scalars(main_tag=MainTagManager.EPOCH_LOSS,
                                 tag_scalar_dict={"train": record.epoch_train_loss},
                                 global_step=record.epoch)

    def on_evaluate_validation_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        pass

    def on_evaluate_validation_epoch_stop(self, trainer: "Trainer", record: Record) -> None:

        logging.info(f"Epoch {record.epoch} evaluate validation loss: {record.epoch_validation_loss:.4f}")
        self._writer.add_scalars(main_tag=MainTagManager.EPOCH_LOSS,
                                 tag_scalar_dict={"validation": record.epoch_validation_loss},
                                 global_step=record.epoch)

    def on_training_complete(self, trainer: "Trainer", record: Record) -> None:
        # 注意这里要 flush 不然可能会导致没有输出
        self._writer.flush()

