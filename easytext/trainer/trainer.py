#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
训练器

Authors: panxu(panxu@baidu.com)
Date:    2020/05/16 00:34:00
"""

import os
import torch
import logging
from typing import List, Union
from tqdm import tqdm
import shutil

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from easytext.model import Model
from easytext.loss import Loss
from easytext.optimizer import OptimizerFactory
from easytext.optimizer import LRSchedulerFactory
from easytext.data.model_collate import ModelInputs
from easytext.metrics import ModelMetricAdapter
from easytext.utils.json_util import json2str
from easytext.utils.nn import cuda_util
from easytext.trainer.metric_tracker import MetricTracker
from easytext.trainer.grad_rescaled import GradRescaled


class Trainer:
    """
    训练器
    """
    _TRAIN = 0
    _EVALUATE = 1

    def __init__(self,
                 serialize_dir: str,
                 num_epoch: int,
                 model: Model,
                 loss: Loss,
                 metrics: ModelMetricAdapter,
                 optimizer_factory: OptimizerFactory,
                 lr_scheduler_factory: LRSchedulerFactory = None,
                 grad_scaled: GradRescaled = None,
                 patient: int = None,
                 num_check_point_keep: int = None,
                 devices: Union[str, List[str]] = None):
        """
        训练器初始化
        :param num_epoch: 训练的 epoch 数量
        :param model: 要训练的模型
        :param loss: 模型的 loss function
        :param metrics: 模型的指标计算
        :param optimizer_factory: 模型的优化器的创建工厂。为什么不直接使用优化器？是因为, 优化器的创建依赖于 model, 所以
        这里传递的参数 optimizer factory, 避免使用者在 trainer 外面生成 optimizer, 导致在 trainer 内 optimizer 依赖于
        model 的参数产生问题。典型问题是: 设置 cuda.
        :param serialize_dir: 训练存储的文件路径
        :param patient: early stopping 的 patient. 如果是 `None`, 将不会进行 early stopping;
        否则, 当前训练的指标超出了 patient 个 epoch 将会 early stopping.
        :param num_check_point_keep: checkpoint 保留的数量。如果是 `None` 则全部保留;
        否则，保留 num_check_point_keep 个checkpoint.
        :param devices: device 字符串, "cuda:0" 或者 "cpu"; 列表类型，多个gpu,  例如 ["cuda:0", "cuda:1"]; 如果是 cpu, 只能是 ["cpu"].
        """
        if devices is None or len(devices) == 0:
            self._devices = [torch.device("cpu")]
        else:

            if isinstance(devices, str):
                self._devices = [torch.device(devices)]
            elif isinstance(devices, list):
                self._devices = [torch.device(device) for device in devices]
            else:
                raise RuntimeError(f"devices type: {type(devices)} 不是 str 或者 list!")

            if len(self._devices) != 1:
                raise RuntimeError(f"目前仅仅支持单 GPU 或 CPU 训练, 设置的 cuda devices 是 {devices}")

        if self._devices[0].type == "cuda":
            self._model = model.cuda(self._devices[0])
        else:
            self._model = model.cpu()

        self._loss = loss
        self._metrics = metrics
        self._optimizer = optimizer_factory.create(model=self._model)

        if lr_scheduler_factory is not None:
            self._lr_scheduler = lr_scheduler_factory.create(optimizer=self.optimizer,
                                                             model=self.model)
        else:
            self._lr_scheduler = None

        self._grad_scaled = grad_scaled

        self._serialize_dir = serialize_dir
        self._metric_tracker = MetricTracker(patient=patient)
        self._num_check_point_keep = num_check_point_keep
        self._num_epoch = num_epoch
        self._current_epoch: int = None

    @property
    def model(self):
        return self._model

    @property
    def loss(self):
        return self._loss

    @property
    def metrics(self):
        return self._metrics

    @property
    def metric_tracker(self):
        return self._metric_tracker

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def serialize_dir(self):
        return self._serialize_dir

    @property
    def num_epoch(self):
        return self._num_epoch

    @property
    def current_epoch(self):
        return self._current_epoch

    def save_checkpoint(self,
                        epoch: int) -> "Trainer":
        """
        保存 checkpoint 到指定的路径下。
        :param serialize_dir: 保存 模型训练相关的文件夹，包括 checkpoint 以及 其他信息
        :param epoch: 保存的 epoch
        :return: self
        """

        # 创建 checkpoint dir

        saved_dir = os.path.join(self._serialize_dir, f"checkpoint_epoch_{epoch}")

        if not os.path.isdir(saved_dir):
            os.makedirs(saved_dir)

        # 模型保存
        model_file_path = os.path.join(saved_dir, "model.pt")
        torch.save(self._model.state_dict(), model_file_path)

        # 优化器保存
        optimizer_file_path = os.path.join(saved_dir, "optimizer.pt")
        torch.save(self._optimizer.state_dict(), optimizer_file_path)

        # 保存lr scheduler

        lr_scheduler_file_path = os.path.join(saved_dir, "lr_scheduler.pt")

        if self._lr_scheduler is None:
            torch.save(None, lr_scheduler_file_path)
        else:
            torch.save(self._lr_scheduler.state_dict(), lr_scheduler_file_path)

        # metric 保存
        # 这里保存的是当前 epoch 的 metric, 在最外面还会保存一份完整的 metric tracker
        metric_file_path = os.path.join(saved_dir, "metric.json")

        metric = self._metric_tracker[epoch]
        with open(metric_file_path, mode="w", encoding="utf-8") as f:
            f.write(f"{json2str(metric, indent=2)}\n")

        # metric tracker 保存
        metric_tracker_file_path = os.path.join(self._serialize_dir, "metric_tracker.json")
        self._metric_tracker.save(metric_tracker_file_path)

        # save best
        if self._metric_tracker.best().epoch == epoch:
            # 保存当前为best
            best_dir = os.path.join(self._serialize_dir, "best")

            if os.path.isdir(best_dir):
                # 已经存在，移动到 backup
                best_bak_dir = os.path.join(self._serialize_dir, "best_bak")

                if os.path.isdir(best_bak_dir):  # 删除 bak
                    shutil.rmtree(best_bak_dir)

                # 将 best 的 备份到 best_bak
                shutil.move(best_dir, best_bak_dir)

            shutil.copytree(saved_dir, best_dir)

        # 删除keep last之外的
        if self._num_check_point_keep is not None:
            for removed_epoch in range(1, epoch - self._num_check_point_keep + 1):
                removed_epoch_dir = os.path.join(self._serialize_dir, f"checkpoint_epoch_{removed_epoch}")

                if os.path.isdir(removed_epoch_dir):
                    shutil.rmtree(removed_epoch_dir)

        return self

    @staticmethod
    def _find_last_epoch(serialize_dir: str):
        """
        寻找最后的 epoch
        :param serialize_dir: serialize 目录
        :return:
        """
        # 找到 last epoch
        last_epoch = None
        for file_name in os.listdir(serialize_dir):

            dir_path = os.path.join(serialize_dir, file_name)

            if os.path.isdir(dir_path):
                parts = file_name.split("_")
                if len(parts) == 3 and parts[0] == "checkpoint" and parts[1] == "epoch":
                    epoch = int(parts[2])

                    if last_epoch is None:
                        last_epoch = epoch
                    else:
                        if epoch > last_epoch:
                            last_epoch = epoch
        return last_epoch

    def load_checkpoint(self,
                        serialize_dir: str) -> "Trainer":
        """
        载入 check point
        :param serialize_dir: 保存的路径
        :return: self
        """

        last_epoch = Trainer._find_last_epoch(serialize_dir=serialize_dir)

        if last_epoch is not None:
            self._current_epoch = last_epoch
            logging.info(f"Load checkpoint, 当前 epoch: {last_epoch}")
            saved_dir = os.path.join(serialize_dir, f"checkpoint_epoch_{last_epoch}")

            model_file_path = os.path.join(saved_dir, "model.pt")
            self._model.load_state_dict(torch.load(model_file_path))

            print(f"last epoch{last_epoch}, loaded: {self._model.state_dict()}")

            optimizer_file_path = os.path.join(saved_dir, "optimizer.pt")
            self._optimizer.load_state_dict(torch.load(optimizer_file_path))

            lr_scheduler_file_path = os.path.join(saved_dir, "lr_scheduler.pt")

            lr_state_dict = torch.load(lr_scheduler_file_path)
            if lr_state_dict is None:
                self._lr_scheduler = None
            else:
                self._lr_scheduler.load_state_dict(lr_state_dict)

            metric_tracker_file_path = os.path.join(serialize_dir, "metric_tracker.json")
            self._metric_tracker = MetricTracker.from_file(metric_tracker_file_path)
        else:
            raise RuntimeError(f"最后保存的epoch数据没有在 {self._serialize_dir} 中找到!")

        return self

    def _train_or_evaluate(self,
                           phrase: int,
                           data_loader: DataLoader) -> float:

        total_loss = 0.
        total_num = 0
        self._metrics.reset()

        if phrase == Trainer._TRAIN:
            self._model.train()
        elif phrase == Trainer._EVALUATE:
            self._model.eval()
        else:
            raise RuntimeError(f"phrase: {phrase} 应该是 {Trainer._TRAIN} 或 {Trainer._EVALUATE}")

        with torch.set_grad_enabled(phrase == Trainer._TRAIN):
            for model_inputs in tqdm(data_loader):
                model_inputs: ModelInputs = model_inputs

                batch_size, batch_inputs, labels = model_inputs.batch_size, \
                                                   model_inputs.model_inputs, \
                                                   model_inputs.labels

                # 设置到 cuda 训练
                if self._devices[0].type == "cuda":  # 仅仅处理 GPU, 默认使用 CPU
                    batch_inputs = cuda_util.cuda(batch_inputs, cuda_device=self._devices[0])
                    labels = cuda_util.cuda(labels, cuda_device=self._devices[0])

                outputs = self._model(**batch_inputs)
                batch_loss: torch.Tensor = self._loss(outputs, labels)

                if phrase == Trainer._TRAIN:
                    self._optimizer.zero_grad()
                    batch_loss.backward()

                    # 反向传播之后修订梯度
                    if self._grad_scaled is not None:
                        self._grad_scaled(self._model)

                    self._optimizer.step()

                total_loss += batch_loss.detach().cpu().item() * batch_size
                total_num += batch_size

                batch_metrics, target_metric = self._metrics(model_outputs=outputs, golden_labels=labels)
                logging.info(f"Epoch: {self._current_epoch}, batch loss: {batch_loss},"
                             f"batch metrics: {json2str(batch_metrics)}, "
                             f"target metric: {json2str(target_metric)}")

        # total_loss = total_loss / total_num 这是合理的 loss, 因为所有的 total_num 是一样的所以，没有一般要再除以一次了
        return total_loss

    def recovery_train(self,
                       train_data_loader: DataLoader,
                       validation_data_loader: DataLoader):
        """
        恢复训练，是指从上次中断的位置重新开始训练
        :param train_data_loader: 训练数据集
        :param validation_data_loader:
        :return:
        """

        self.load_checkpoint(self.serialize_dir)
        self._train(train_data_loader=train_data_loader,
                    validation_data_loader=validation_data_loader)

    def evaluate(self,
                 validation_data_loader: DataLoader) -> float:
        """
        评估验证集
        :param validation_data_loader: 验证集data loader
        :return: loss 结果
        """
        return self._train_or_evaluate(phrase=Trainer._EVALUATE,
                                       data_loader=validation_data_loader)

    def _is_serialize_empty(self):
        """
        判断 serialize dir 是否是空的，会忽略 隐藏文件
        :return: True: 空文件夹; False: 非空文件夹
        """
        if not os.path.isdir(self._serialize_dir):
            raise RuntimeError(f"保存路径是无效的路径: {self._serialize_dir} ")

        is_empty = True
        for name in os.listdir(self._serialize_dir):
            if not name.startswith("."):
                is_empty = False
                break
        return is_empty

    def train(self, train_data_loader: DataLoader,
              validation_data_loader: DataLoader) -> None:
        if not self._is_serialize_empty():
            raise RuntimeError(f"新训练，请清空保存文件件: {self._serialize_dir}")

        return self._train(train_data_loader=train_data_loader,
                           validation_data_loader=validation_data_loader)

    def _train(self,
               train_data_loader: DataLoader,
               validation_data_loader: DataLoader) -> None:
        """
        模型训练
        :param train_data_loader: 训练集 data loader
        :param validation_data_loader: 验证集 data loader
        :return:
        """

        if self._current_epoch is None:
            start_epoch = 1
        else:
            start_epoch = self._current_epoch

        for epoch in range(start_epoch, self._num_epoch + 1):

            self._current_epoch = epoch

            logging.info(f"Start train epoch: {self._current_epoch}")

            train_loss = self._train_or_evaluate(phrase=Trainer._TRAIN,
                                                 data_loader=train_data_loader)

            # 输出metrics
            train_metric_dict, train_target_metric = self._metrics.metric
            logging.info(
                f"Train epoch: {epoch}, loss: {train_loss}, target metric: {train_target_metric.name}:{train_target_metric.value} "
                f"metrics: {json2str(train_metric_dict)}")

            evaluate_loss = self.evaluate(validation_data_loader=validation_data_loader)
            validation_metric_dict, validation_target_metric = self._metrics.metric

            self._metric_tracker.add_metric(epoch=epoch,
                                            train_metric=train_metric_dict,
                                            train_model_target_metric=train_target_metric,
                                            validation_metric=validation_metric_dict,
                                            validation_model_target_metric=validation_target_metric)
            logging.info(
                f"Evaluate Valid epoch: {epoch}, loss: {evaluate_loss}, "
                f"target metric: {validation_target_metric.name}:{validation_target_metric.value} "
                f"metrics: {json2str(validation_metric_dict)}")

            # 设置 lr scheduler
            # 注意这样设置会有点问题，对于某些 scheduler step 需要参数, 例如: ReduceLROnPlateau
            # 这种情况，暂时不处理, 如果处理需要重新重构，或者简单实用 isinstance ReduceLROnPlateau 来处理
            # 这里简单实用 isinstance 处理。必须指出，ReduceLROnPlateau 的基类是 object, 这也是多少有些问题。

            if self._lr_scheduler is not None:
                if isinstance(self._lr_scheduler, ReduceLROnPlateau):
                    self._lr_scheduler.step(metrics=evaluate_loss, epoch=epoch)
                else:
                    self._lr_scheduler.step(epoch=epoch)

            self.save_checkpoint(epoch=epoch)

            if self._metric_tracker.early_stopping(epoch):
                logging.info(f"Epoch: {epoch}, early stopping!")
                break
