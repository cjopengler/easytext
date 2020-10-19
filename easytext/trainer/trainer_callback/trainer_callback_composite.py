#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
所有 callback 来这里添加

Authors: PanXu
Date:    2020/10/16 17:39:00
"""
from typing import Set

from easytext.trainer import Record
from easytext.trainer.trainer_callback import TrainerCallback


class TrainerCallbackComposite(TrainerCallback):
    """
    所有 trainer callback composite
    """

    def __init__(self):
        self._callbacks: Set[TrainerCallback] = set()

    def add_callback(self, callback: TrainerCallback) -> "TrainerCallbackComposite":
        self._callbacks.add(callback)
        return self

    def remove_callback(self, callback: TrainerCallback) -> "TrainerCallbackComposite":

        if callback in self._callbacks:
            self._callbacks.remove(callback)
        return self

    def on_train_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        for callback in self._callbacks:
            callback.on_train_epoch_start(trainer=trainer,
                                          record=record)

    def on_train_epoch_stop(self, trainer: "Trainer", record: Record) -> None:
        for callback in self._callbacks:
            callback.on_train_epoch_stop(trainer=trainer,
                                         record=record)

    def on_evaluate_validation_epoch_start(self, trainer: "Trainer", record: Record) -> None:
        for callback in self._callbacks:
            callback.on_evaluate_validation_epoch_start(trainer=trainer,
                                                        record=record)

    def on_evaluate_validation_epoch_stop(self, trainer: "Trainer", record: Record) -> None:
        for callback in self._callbacks:
            callback.on_evaluate_validation_epoch_stop(trainer=trainer,
                                                       record=record)

    def on_training_complete(self, trainer: "Trainer", record: Record) -> None:
        for callback in self._callbacks:
            callback.on_training_complete(trainer=trainer,
                                          record=record)
