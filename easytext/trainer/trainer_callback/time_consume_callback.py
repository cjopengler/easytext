#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
时间消耗 callback

Authors: PanXu
Date:    2020/10/16 15:48:00
"""
import logging
import time

from torch.utils.tensorboard import SummaryWriter

from easytext.trainer import Record
from easytext.trainer.trainer_callback import TrainerCallback
from easytext.trainer.tensorboard_manager import MainTagManager


class TimeConsumeCallback(TrainerCallback):
    """
    时间消耗 callback 用来统计不同阶段的时间, 时间单位是 毫秒 ms
    """

    def __init__(self, tensorboard_summary_writer: SummaryWriter):
        self._writer = tensorboard_summary_writer
        self._train_epoch_start_time: int = None
        self._evaluate_epoch_start_time: int = None

    def on_train_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        self._train_epoch_start_time = time.time()

    def on_train_epoch_stop(self, trainer: "Trainer", record: Record) -> None:
        stop_time = time.time()
        used_time = stop_time - self._train_epoch_start_time
        used_time = int(1000 * used_time)
        logging.info(f"Train epoch {record.epoch} used: {used_time}ms")
        self._writer.add_scalars(main_tag=MainTagManager.EPOCH_USED_TIME,
                                 tag_scalar_dict={"train": used_time},
                                 global_step=record.epoch)

    def on_evaluate_validation_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        self._evaluate_epoch_start_time = time.time()

    def on_evaluate_validation_epoch_stop(self, trainer: "Trainer", record: Record) -> None:
        stop_time = time.time()
        used_time = stop_time - self._evaluate_epoch_start_time
        used_time = int(1000 * used_time)
        logging.info(f"Evaluate validation epoch {record.epoch} used: {used_time}ms")
        self._writer.add_scalars(main_tag=MainTagManager.EPOCH_USED_TIME,
                                 tag_scalar_dict={"validation": used_time},
                                 global_step=record.epoch)

    def on_training_complete(self, trainer: "Trainer", record: Record) -> None:
        # 注意这里要 flush 不然可能会导致没有输出
        self._writer.flush()

