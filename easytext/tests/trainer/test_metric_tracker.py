#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
测试 metric tracker

Authors: panxu(panxu@baidu.com)
Date:    2020/05/28 14:46:00
"""
import os
from easytext.tests import ASSERT
from easytext.tests import ROOT_PATH

from easytext.trainer import MetricTracker
from easytext.metrics import ModelTargetMetric

METRICS = [{"epoch": 1,
            "train_metric": {"acc": 0.81},
            "train_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.81),
            "validation_metric": {"acc": 0.46},
            "validation_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.46)},
           {"epoch": 2,
            "train_metric": {"acc": 0.83},
            "train_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.83),
            "validation_metric": {"acc": 0.48},
            "validation_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.48)},
           {"epoch": 3,
            "train_metric": {"acc": 0.85},
            "train_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.85),
            "validation_metric": {"acc": 0.60},
            "validation_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.60)},
           {"epoch": 4,
            "train_metric": {"acc": 0.89},
            "train_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.89),
            "validation_metric": {"acc": 0.44},
            "validation_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.44)},
           {"epoch": 5,
            "train_metric": {"acc": 0.92},
            "train_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.92),
            "validation_metric": {"acc": 0.39},
            "validation_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.39)}
           ]


def test_metric_tracker_best():
    """
    测试 metric tracker
    :return:
    """
    metric_tracker = MetricTracker(patient=None)

    for metric in METRICS:
        metric_tracker.add_metric(**metric)

    expect = {"epoch": 3,
              "train_metric": {"acc": 0.85},
              "train_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.85),
              "validation_metric": {"acc": 0.60},
              "validation_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.60)}

    best = metric_tracker.best()
    ASSERT.assertEqual(expect["epoch"], best.epoch)

    ASSERT.assertDictEqual(expect["train_metric"], best.train_metric)
    ASSERT.assertDictEqual(expect["validation_metric"], best.validation_metric)
    ASSERT.assertEqual(expect["train_model_target_metric"].name, best.train_model_target_metric.name)
    ASSERT.assertEqual(expect["train_model_target_metric"].value, best.train_model_target_metric.value)
    ASSERT.assertEqual(expect["validation_model_target_metric"].name, best.validation_model_target_metric.name)
    ASSERT.assertEqual(expect["validation_model_target_metric"].value, best.validation_model_target_metric.value)


def test_metric_tracker_patient():
    metric_tracker = MetricTracker(patient=1)

    for metric in METRICS:
        metric_tracker.add_metric(**metric)

        if metric["epoch"] > 4:
            ASSERT.assertTrue(metric_tracker.early_stopping(metric["epoch"]))
        else:
            ASSERT.assertFalse(metric_tracker.early_stopping(metric["epoch"]))

        if metric_tracker.early_stopping(metric["epoch"]):
            break

    expect = {"epoch": 3,
              "train_metric": {"acc": 0.85},
              "train_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.85),
              "validation_metric": {"acc": 0.60},
              "validation_model_target_metric": ModelTargetMetric(metric_name="acc", metric_value=0.60)}

    best = metric_tracker.best()
    ASSERT.assertEqual(expect["epoch"], best.epoch)

    ASSERT.assertDictEqual(expect["train_metric"], best.train_metric)
    ASSERT.assertDictEqual(expect["validation_metric"], best.validation_metric)
    ASSERT.assertEqual(expect["train_model_target_metric"].name, best.train_model_target_metric.name)
    ASSERT.assertEqual(expect["train_model_target_metric"].value, best.train_model_target_metric.value)
    ASSERT.assertEqual(expect["validation_model_target_metric"].name, best.validation_model_target_metric.name)
    ASSERT.assertEqual(expect["validation_model_target_metric"].value, best.validation_model_target_metric.value)


def test_metric_tracker_save_and_load():
    metric_tracker = MetricTracker(patient=1)

    for metric in METRICS:
        metric_tracker.add_metric(**metric)

        if metric["epoch"] > 4:
            ASSERT.assertTrue(metric_tracker.early_stopping(metric["epoch"]))
        else:
            ASSERT.assertFalse(metric_tracker.early_stopping(metric["epoch"]))

        if metric_tracker.early_stopping(metric["epoch"]):
            break

    saved_file_path = os.path.join(ROOT_PATH, "data/easytext/tests/trainer/metric_tracker.json")

    metric_tracker.save(saved_file_path)

    loaded_metric_tracker = MetricTracker.from_file(saved_file_path)

    best = metric_tracker.best()
    loaded_best = loaded_metric_tracker.best()
    ASSERT.assertEqual(best.epoch, loaded_best.epoch)

    ASSERT.assertDictEqual(best.train_metric, loaded_best.train_metric)
    ASSERT.assertDictEqual(best.validation_metric, loaded_best.validation_metric)
    ASSERT.assertEqual(best.train_model_target_metric.name, loaded_best.train_model_target_metric.name)
    ASSERT.assertEqual(best.train_model_target_metric.value, loaded_best.train_model_target_metric.value)
    ASSERT.assertEqual(best.validation_model_target_metric.name, loaded_best.validation_model_target_metric.name)
    ASSERT.assertEqual(best.validation_model_target_metric.value, loaded_best.validation_model_target_metric.value)