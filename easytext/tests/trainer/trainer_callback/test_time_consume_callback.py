#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 time consume callback,

因为只能通过 tensorboard 查看，所以无法使用 assert

Authors: PanXu
Date:    2020/10/16 16:32:00
"""
import time
import os
import numpy as np

from easytext.utils import log_util

from easytext.trainer import Record
from easytext.trainer.trainer_callback import TimeConsumeCallback

log_util.config()


def test_time_consume_callback(summary_writer):

    callback = TimeConsumeCallback(tensorboard_summary_writer=summary_writer)

    for i in range(1, 10):
        record = Record()
        record.epoch = i

        callback.on_train_epoch_start(trainer=None, record=record)
        time.sleep(0.02)
        callback.on_train_epoch_stop(trainer=None, record=record)

        callback.on_evaluate_validation_epoch_start(trainer=None, record=record)
        time.sleep(0.01)
        callback.on_evaluate_validation_epoch_stop(trainer=None, record=record)

    summary_writer.flush()