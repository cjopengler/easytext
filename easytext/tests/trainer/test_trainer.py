#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
测试 保存和载入 checkpoint

Authors: panxu(panxu@baidu.com)
Date:    2020/05/29 11:11:00
"""
import json
import os
import shutil
import logging
from typing import Iterable, List, Dict, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from easytext.data import Instance, LabelVocabulary
from easytext.data.model_collate import ModelInputs
from easytext.data.model_collate import ModelCollate
from easytext.model import Model
from easytext.model import ModelOutputs
from easytext.loss import Loss
from easytext.optimizer import OptimizerFactory
from easytext.trainer import Trainer
from easytext.metrics import AccMetric, ModelMetricAdapter, ModelTargetMetric
from easytext.utils import log_util
from easytext.utils.json_util import json2str
from easytext.label_decoder import ModelLabelDecoder
from easytext.label_decoder import MaxLabelIndexDecoder

from easytext.tests import ASSERT
from easytext.tests import ROOT_PATH


log_util.config()


class _DemoDataset(Dataset):
    """
    测试用的数据集
    """

    def __init__(self):
        self._instances: List[Instance] = list()

        max_num = 100
        for i in range(0, 50):
            instance1 = Instance()
            instance1["x"] = i

            instance2 = Instance()
            instance2["x"] = max_num - i

            self._instances.append(instance1)
            self._instances.append(instance2)

    def __len__(self):
        return len(self._instances)

    def __getitem__(self, index):
        return self._instances[index]


class _DemoCollate(ModelCollate):
    """
    测试用的 collate
    """

    def __call__(self, instances: Iterable[Instance]) -> ModelInputs:

        x = list()
        labels = list()
        for instance in instances:

            x_data = instance["x"]
            x.append(torch.tensor([x_data], dtype=torch.float))

            if x_data - 50 > 0:
                labels.append(1)
            else:
                labels.append(0)

        x = torch.stack(x)

        batch_size = x.size(0)
        ASSERT.assertEqual(x.dim(), 2)
        ASSERT.assertListEqual([batch_size, 1], [x.size(0), x.size(1)])

        labels = torch.tensor(labels)
        ASSERT.assertEqual(labels.dim(), 1)
        ASSERT.assertEqual(batch_size, labels.size(0))

        model_inputs = ModelInputs(batch_size=batch_size,
                                   model_inputs={"x": x},
                                   labels=labels)

        return model_inputs


class _DemoOutputs(ModelOutputs):

    def __init__(self, logits: torch.Tensor):
        super().__init__(logits=logits)


class _DemoLabelDecoder(ModelLabelDecoder):

    def __init__(self):
        super().__init__()
        self._label_index_decoder = MaxLabelIndexDecoder()

    def decode_label_index(self, model_outputs: ModelOutputs) -> torch.LongTensor:
        model_outputs: _DemoOutputs = model_outputs
        return self._label_index_decoder(model_outputs.logits)

    def decode_label(self, model_outputs: ModelOutputs, label_indices: torch.LongTensor) -> List:
        return [label_index.item() for label_index in label_indices]


class _DemoMetric(ModelMetricAdapter):

    def __init__(self):
        super().__init__()
        self._acc = AccMetric()
        self._label_decoder = _DemoLabelDecoder()

    def __call__(self, model_outputs: _DemoOutputs, golden_labels: Tensor) -> Tuple[Dict, ModelTargetMetric]:
        model_outputs: _DemoOutputs = model_outputs
        label_indices = self._label_decoder.decode_label_index(model_outputs=model_outputs)
        acc = self._acc(prediction_labels=label_indices, gold_labels=golden_labels, mask=None)

        target = ModelTargetMetric(AccMetric.ACC, acc[AccMetric.ACC])
        return acc, target

    @property
    def metric(self) -> Tuple[Dict, ModelTargetMetric]:
        acc = self._acc.metric

        target = ModelTargetMetric(AccMetric.ACC, acc[AccMetric.ACC])
        return acc, target

    def reset(self) -> "_DemoMetric":
        self._acc.reset()
        return self


class _DemoOptimizerFactory(OptimizerFactory):

    def create(self, model: Model) -> "Optimizer":
        return torch.optim.Adam(params=model.parameters(), lr=1e-1)


class ModelDemo(Model):

    def __init__(self):
        super().__init__()
        self.feed_forward = torch.nn.Linear(in_features=1, out_features=2)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, x: torch.Tensor) -> _DemoOutputs:

        # print(self.feed_forward.weight, self.feed_forward.bias)
        logits = self.feed_forward(x)

        demo_outputs = _DemoOutputs(logits=logits)
        return demo_outputs


class _DemoLoss(Loss):

    def __init__(self):
        super().__init__()
        self._loss = torch.nn.CrossEntropyLoss()

    def __call__(self, model_outputs: ModelOutputs, golden_label: torch.Tensor) -> torch.Tensor:
        model_outputs: _DemoOutputs = model_outputs

        return self._loss(model_outputs.logits, golden_label)


def _run_train(devices: List[str] = None):
    serialize_dir = os.path.join(ROOT_PATH, "data/easytext/tests/trainer/save_and_load")

    if os.path.isdir(serialize_dir):
        shutil.rmtree(serialize_dir)

    os.makedirs(serialize_dir)

    model = ModelDemo()

    optimizer_factory = _DemoOptimizerFactory()

    loss = _DemoLoss()
    metric = _DemoMetric()

    trainer = Trainer(num_epoch=100,
                      model=model,
                      loss=loss,
                      metrics=metric,
                      optimizer_factory=optimizer_factory,
                      serialize_dir=serialize_dir,
                      patient=20,
                      num_check_point_keep=25,
                      devices=devices
                      )

    train_dataset = _DemoDataset()

    train_data_loader = DataLoader(dataset=train_dataset,
                                   collate_fn=_DemoCollate(),
                                   batch_size=200,
                                   shuffle=False,
                                   num_workers=0)

    validation_data_loader = DataLoader(dataset=train_dataset,
                                        collate_fn=_DemoCollate(),
                                        batch_size=200,
                                        shuffle=False,
                                        num_workers=0)

    trainer.train(train_data_loader=train_data_loader,
                  validation_data_loader=validation_data_loader)

    expect_model_state_dict = json.loads(json2str(trainer.model.state_dict()))
    expect_optimizer_state_dict = json.loads(json2str(trainer.optimizer.state_dict()))
    expect_current_epoch = trainer.current_epoch
    expect_num_epoch = trainer.num_epoch
    expect_metric = trainer.metrics.metric[0]
    expect_metric_tracker = json.loads(json2str(trainer.metric_tracker))

    trainer.load_checkpoint(serialize_dir=serialize_dir)

    loaded_model_state_dict = json.loads(json2str(trainer.model.state_dict()))
    loaded_optimizer_state_dict = json.loads(json2str(trainer.optimizer.state_dict()))
    current_epoch = trainer.current_epoch
    num_epoch = trainer.num_epoch
    metric = trainer.metrics.metric[0]
    metric_tracker = json.loads(json2str(trainer.metric_tracker))

    ASSERT.assertDictEqual(expect_model_state_dict, loaded_model_state_dict)
    ASSERT.assertDictEqual(expect_optimizer_state_dict, loaded_optimizer_state_dict)
    ASSERT.assertEqual(expect_current_epoch, current_epoch)
    ASSERT.assertEqual(expect_num_epoch, num_epoch)
    ASSERT.assertDictEqual(expect_metric, metric)
    ASSERT.assertDictEqual(expect_metric_tracker, metric_tracker)


def test_trainer_save_and_load_cpu():
    """
    测试  trainer 保存和载入
    :return:
    """
    _run_train(devices="cpu")


def test_trainer_save_and_load_cpu_with_none_parameter():
    """
    测试  trainer cpu 保存和载入, 不设置任何参数，默认使用cpu
    :return:
    """
    _run_train()


def test_trainer_save_load_gpu():

    if torch.cuda.is_available():
        cuda_devices = ["cuda:0"]
        _run_train(devices=cuda_devices)
    else:
        logging.warning("由于没有GPU，忽略这个case测试")


