#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
基础的 trainer callback

Authors: PanXu
Date:    2020/10/19 11:00:00
"""

from torch.utils.tensorboard import SummaryWriter

from easytext.trainer import Record
from easytext.trainer.trainer_callback import TimeConsumeCallback
from easytext.trainer.trainer_callback import LossCallback
from easytext.trainer.trainer_callback import MetricCallback
from easytext.trainer.trainer_callback import TrainerCallbackComposite


class BasicTrainerCallbackComposite(TrainerCallbackComposite):
    """
    basic trainer callback
    """

    def __init__(self, tensorboard_log_dir: str):
        """
        初始化
        :param tensorboard_log_dir: tensorboard 的 log dir
        """
        super().__init__()
        self._summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)

        self._callbacks.add(TimeConsumeCallback(tensorboard_summary_writer=self._summary_writer))
        self._callbacks.add(LossCallback(tensorboard_summary_writer=self._summary_writer))
        self._callbacks.add(MetricCallback(tensorboard_summary_writer=self._summary_writer))

    def on_training_complete(self, trainer: "Trainer", record: Record) -> None:
        self._summary_writer.close()
