#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
Metric 跟踪器

Authors: panxu(panxu@baidu.com)
Date:    2020/05/28 09:01:00
"""
import json
from typing import Dict

from easytext.metrics import ModelTargetMetric
from easytext.utils.json_util import json2str


class MetricTrackerItem:
    """
    Metric Tracker Item
    """

    def __init__(self, epoch: int,
                 train_metric: Dict,
                 train_model_target_metric: ModelTargetMetric,
                 validation_metric: Dict,
                 validation_model_target_metric: ModelTargetMetric):
        """
        :param epoch: 当前 epoch
        :param train_metric: 当前 epoch的模型 metric
        :param train_model_target_metric: 当前模型的 target metric
        :param validation_metric: 当前 epoch的模型 metric
        :param validation_model_target_metric: 当前模型的 target metric
        """
        self.epoch = epoch
        self.train_metric = train_metric
        self.train_model_target_metric = train_model_target_metric
        self.validation_metric = validation_metric
        self.validation_model_target_metric = validation_model_target_metric


class MetricTracker:
    """
    Metric 跟踪器
    """

    def __init__(self, patient: int = None):
        self._best_epoch = None
        self.patient = patient

        # 会将每一个 epoch 产生的都记录下来
        self.metric_tracker_dict: Dict[int, MetricTrackerItem] = dict()

    def _calculate_best(self, epoch: int) -> "MetricTracker":
        """
        比较当前 epoch 与 best, 返回 best epoch
        :param epoch: 当前 epoch
        :return:
        """

        if self._best_epoch is None:
            self._best_epoch = epoch
        else:
            best_metric_value = self.metric_tracker_dict[self._best_epoch].validation_model_target_metric.value

            current_metric_value = self.metric_tracker_dict[epoch].validation_model_target_metric.value

            if current_metric_value > best_metric_value:
                self._best_epoch = epoch

        return self

    def add_metric(self,
                   epoch: int,
                   train_metric: Dict,
                   train_model_target_metric: ModelTargetMetric,
                   validation_metric: Dict,
                   validation_model_target_metric: ModelTargetMetric) -> "MetricTracker":
        """
        添加 metric
        :param epoch: 当前 epoch
        :param train_metric: 当前 epoch的模型 metric
        :param train_model_target_metric: 当前模型的 target metric
        :param validation_metric: 当前 epoch的模型 metric
        :param validation_model_target_metric: 当前模型的 target metric
        :return:
        """
        item = MetricTrackerItem(epoch=epoch,
                                 train_metric=train_metric,
                                 train_model_target_metric=train_model_target_metric,
                                 validation_metric=validation_metric,
                                 validation_model_target_metric=validation_model_target_metric)
        self.metric_tracker_dict[epoch] = item

        self._calculate_best(epoch)
        return self

    def best(self):
        """
        返回最好的
        :return:
        """
        if self._best_epoch is not None:
            return self.metric_tracker_dict[self._best_epoch]
        return None

    def __getitem__(self, epoch: int):
        return self.metric_tracker_dict[epoch]

    def __len__(self):
        return len(self.metric_tracker_dict)

    def save(self, file_path: str):
        """
        将 metric tracker 保存到文件
        :param file_path:
        :return:
        """
        with open(file_path, mode="w", encoding="utf-8") as f:
            f.write(f"{json2str(self, indent=2)}\n")
        return self

    @classmethod
    def from_file(cls, file_path: str):
        """
        从文件中载入  metric tracker
        :param file_path: 文件路径
        :return:
        """

        with open(file_path, mode="r", encoding="utf-8") as f:
            metric_tracker = cls()
            data_dict = json.load(f)
            metric_tracker.patient = data_dict["patient"]
            metric_tracker._best_epoch = data_dict["_best_epoch"]

            for epoch, item in data_dict["metric_tracker_dict"].items():
                epoch = int(epoch)
                train_metric = item["train_metric"]

                validation_metric = item["validation_metric"]

                train_model_target_metric = ModelTargetMetric(
                    metric_name=item["train_model_target_metric"]["_metric_name"],
                    metric_value=item["train_model_target_metric"]["_metric_value"])

                validation_model_target_metric = ModelTargetMetric(
                    metric_name=item["validation_model_target_metric"]["_metric_name"],
                    metric_value=item["validation_model_target_metric"]["_metric_value"])

                metric_tracker.metric_tracker_dict[epoch] = MetricTrackerItem(
                    epoch=item["epoch"],
                    train_metric=train_metric,
                    train_model_target_metric=train_model_target_metric,
                    validation_metric=validation_metric,
                    validation_model_target_metric=validation_model_target_metric)

            return metric_tracker

    def early_stopping(self, epoch: int) -> bool:
        """
        early stopping
        :return True: 停止训练; False: 不会停止训练:
        """
        is_stopping = False

        if self.patient is not None:
            is_stopping = (epoch - self._best_epoch) > self.patient

        return is_stopping
