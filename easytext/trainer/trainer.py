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
from tqdm import tqdm
import shutil

from typing import Optional, Union, List

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as TorchDist
from torch.distributed import ReduceOp

from easytext.model import Model
from easytext.loss import Loss
from easytext.optimizer import OptimizerFactory
from easytext.optimizer import LRSchedulerFactory
from easytext.data.model_collate import ModelInputs
from easytext.metrics import ModelMetricAdapter
from easytext.utils.json_util import json2str
from easytext.utils.nn import cuda_util
from easytext.utils.distributed.distributed_util import DistributedFuncWrapper
from easytext.utils.distributed import Sync
from easytext.trainer.metric_tracker import MetricTracker
from easytext.trainer.grad_rescaled import GradRescaled
from easytext.trainer import Record
from easytext.trainer.trainer_callback import TrainerCallback
from easytext.distributed import Distributed
from easytext.distributed import DistributedDataParallelParameter, ProcessGroupParameter


class NerModelOutputs:
    """
    Ner Model Outputs
    """

    def __init__(self, logits, mask, crf):
        """
        Ner 模型的输出
        :param logits: logits 输出
        :param mask: mask
        :param crf: 模型中的 crf 输出出来，用来进行 loss 以及 viterbi 解码
        """
        self.logits = logits
        self.mask = mask
        self.crf = crf


class Trainer(TrainerCallback, Distributed):
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
                 device: torch.device,
                 is_distributed: bool,
                 lr_scheduler_factory: LRSchedulerFactory = None,
                 grad_scaled: GradRescaled = None,
                 patient: int = None,
                 num_check_point_keep: int = None,
                 trainer_callback: Union[TrainerCallback, List[TrainerCallback], None] = None,
                 distributed_data_parallel_parameter: DistributedDataParallelParameter = None
                 ):
        """
        训练器初始化
        :param num_epoch: 训练的 epoch 数量
        :param model: 要训练的模型
        :param loss: 模型的 loss function
        :param metrics: 模型的指标计算
        :param optimizer_factory: 模型的优化器的创建工厂。为什么不直接使用优化器？是因为, 优化器的创建依赖于 model, 所以
        这里传递的参数 optimizer factory, 避免使用者在 trainer 外面生成 optimizer, 导致在 trainer 内 optimizer 依赖于
        model 的参数产生问题。典型问题是: 设置 cuda.
        :param device: 训练时所依赖的 device
        :param is_distributed: 当前 trainer 是否是在多 GPU 环境下使用
        :param serialize_dir: 训练存储的文件路径
        :param patient: early stopping 的 patient. 如果是 `None`, 将不会进行 early stopping;
        否则, 当前训练的指标超出了 patient 个 epoch 将会 early stopping.
        :param num_check_point_keep: checkpoint 保留的数量。如果是 `None` 则全部保留;
        否则，保留 num_check_point_keep 个checkpoint.
        :param trainer_callback: 训练中的回调。可以是 List, 如果是 List, 怎按照顺序逐个执行.
            当 Trainer.is_distributed == True, 会去查看 trainer_back.is_distributed, True 表示
        :param distributed_data_parallel_parameter: DistributedDataParallel 用到的参数, 目前只支持设置 find_unused_parameters
        """

        self._device = device
        self._loss = loss
        self._metrics = metrics
        self._optimizer_factory = optimizer_factory

        self._grad_scaled = grad_scaled

        self._serialize_dir = serialize_dir
        self._metric_tracker = MetricTracker(patient=patient)
        self._num_check_point_keep = num_check_point_keep
        self._num_epoch = num_epoch
        self._current_epoch: Optional[int] = None
        self._trainer_callback = trainer_callback

        self._is_distributed = is_distributed

        if self.is_distributed:
            self._distributed_func_wrapper = DistributedFuncWrapper(dst_rank=0)

            self._ddp = distributed_data_parallel_parameter \
                        or DistributedDataParallelParameter(find_unused_parameters=False)
        else:
            self._distributed_func_wrapper = DistributedFuncWrapper(dst_rank=None)

        self._model = model
        if self.is_distributed:

            assert self._device.type != "cpu", f"多 GPU 训练, device 不能是 cpu"

            torch.cuda.set_device(self._device)
            self._model.cuda(self._device)
            self._optimizer = optimizer_factory.create(self._model)

            self._model = DistributedDataParallel(module=self._model,
                                                  device_ids=[self._device],
                                                  output_device=self._device,
                                                  find_unused_parameters=self._ddp.find_unused_parameters)
        else:
            self._model = self._model.to(self._device)
            self._optimizer = self._optimizer_factory.create(self._model)

        if lr_scheduler_factory is not None:
            self._lr_scheduler = lr_scheduler_factory.create(optimizer=self.optimizer,
                                                             model=self.model)
        else:
            self._lr_scheduler = None

        self._check_distributed()

    def _check_distributed(self):
        """
        检查 metric 的合法性, 如果非法会抛出异常
        :return:
        """

        if self._trainer_callback is not None:
            assert self._trainer_callback.is_distributed == self.is_distributed, \
                f"当前 trainer_callback is_distributed: {self._trainer_callback.is_distributed} " \
                f"与 trainer is_distributed:{self.is_distributed} 不相等"

    @property
    def is_distributed(self) -> bool:
        return self._is_distributed

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

            if self.is_distributed:

                if self._distributed_func_wrapper is not None \
                        and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                    logging.info(f"Load checkpoint, 当前 epoch: {last_epoch}")
            else:
                logging.info(f"Load checkpoint, 当前 epoch: {last_epoch}")

            saved_dir = os.path.join(serialize_dir, f"checkpoint_epoch_{last_epoch}")

            model_file_path = os.path.join(saved_dir, "model.pt")
            self._model.load_state_dict(torch.load(model_file_path))

            if self.is_distributed:

                if self._distributed_func_wrapper is not None \
                        and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                    logging.info(f"last epoch{last_epoch}, loaded: {self._model.state_dict()}")
            else:
                logging.info(f"last epoch{last_epoch}, loaded: {self._model.state_dict()}")

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

        self._metrics.reset()

        if phrase == Trainer._TRAIN:
            self._model.train()
        elif phrase == Trainer._EVALUATE:
            self._model.eval()
        else:
            raise RuntimeError(f"phrase: {phrase} 应该是 {Trainer._TRAIN} 或 {Trainer._EVALUATE}")

        with torch.set_grad_enabled(phrase == Trainer._TRAIN):

            if self.is_distributed:

                tqdm_disable = True

                if self._distributed_func_wrapper is not None \
                    and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                    tqdm_disable = False
            else:
                tqdm_disable = False

            for model_inputs in tqdm(data_loader, disable=tqdm_disable):
                model_inputs: ModelInputs = model_inputs

                batch_size, batch_inputs, labels \
                    = model_inputs.batch_size, model_inputs.model_inputs, model_inputs.labels

                # 设置到 cuda 训练
                if self._device.type == "cuda":  # 仅仅处理 GPU, 默认使用 CPU
                    batch_inputs = cuda_util.cuda(batch_inputs, cuda_device=self._device)
                    labels = cuda_util.cuda(labels, cuda_device=self._device)

                outputs = self._model(**batch_inputs)

                batch_loss: torch.Tensor = self._loss(outputs, labels)

                if phrase == Trainer._TRAIN:
                    self._optimizer.zero_grad()
                    batch_loss.backward()

                    # 反向传播之后修订梯度
                    if self._grad_scaled is not None:
                        self._grad_scaled(self._model)

                    self._optimizer.step()

                total_loss += batch_loss.detach().item() * batch_size

                batch_metrics, target_metric = self._metrics(model_outputs=outputs, golden_labels=labels)

                if self.is_distributed:

                    if self._distributed_func_wrapper is not None \
                            and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                        logging.info(f"Epoch: {self._current_epoch}, batch loss: {batch_loss}"
                                     f"batch metrics: {json2str(batch_metrics)}, "
                                     f"target metric: {json2str(target_metric)}")
                else:
                    logging.info(f"Epoch: {self._current_epoch}, batch loss: {batch_loss},"
                                 f"batch metrics: {json2str(batch_metrics)}, "
                                 f"target metric: {json2str(target_metric)}")

        # total_loss = total_loss / total_num 这是合理的 loss, 因为所有的 total_num 是一样的所以，没有必要再除以一次了
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

        if self.is_distributed:
            TorchDist.barrier()

        self._train(train_data_loader=train_data_loader,
                    validation_data_loader=validation_data_loader)

        if self.is_distributed:
            TorchDist.barrier()

    def evaluate(self,
                 validation_data_loader: DataLoader) -> float:
        """
        评估验证集
        :param validation_data_loader: 验证集data loader
        :return: loss 结果, 以及当前数量
        """
        loss = self._train_or_evaluate(phrase=Trainer._EVALUATE,
                                       data_loader=validation_data_loader)
        if self.is_distributed:
            TorchDist.barrier()

        return loss

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

    def _check_data_loader_validity(self, data_loader: DataLoader):
        """
        检查 data loader 是否有效
        :param data_loader: data loader
        :return:
        """

        if self.is_distributed:
            assert isinstance(data_loader.sampler, DistributedSampler), \
                f"data_loader.sampler 必须是 DistributedSampler 实例"

    def train(self,
              train_data_loader: DataLoader,
              validation_data_loader: DataLoader) -> None:
        """
        模型训练
        :param train_data_loader: 训练集 data loader
        :param validation_data_loader: 验证集 data loader
        :return:
        """

        if not self._is_serialize_empty():
            raise RuntimeError(f"新训练，请清空保存文件件: {self._serialize_dir}")

        if self.is_distributed:
            TorchDist.barrier()

        self._check_data_loader_validity(train_data_loader)
        self._check_data_loader_validity(validation_data_loader)

        self._train(train_data_loader=train_data_loader,
                    validation_data_loader=validation_data_loader)

        if self.is_distributed:
            TorchDist.barrier()

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

        record = Record()

        for epoch in range(start_epoch, self._num_epoch + 1):
            record.epoch = epoch

            self._current_epoch = epoch

            if self.is_distributed:

                if self._distributed_func_wrapper is not None \
                        and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                    logging.info(f"Start train epoch: {self._current_epoch}")
            else:
                logging.info(f"Start train epoch: {self._current_epoch}")

            self.on_train_epoch_start(trainer=self, record=record)

            train_loss = self._train_or_evaluate(phrase=Trainer._TRAIN, data_loader=train_data_loader)

            if self.is_distributed:
                train_loss = Sync.sync(train_loss, device=self._device, op=ReduceOp.SUM)

            record.epoch_train_loss = train_loss

            if self.is_distributed:
                sync_data, op = self._metrics.to_synchronized_data()
                sync_data = Sync.sync(sync_data, device=self._device, op=op)

                self._metrics.from_synchronized_data(sync_data=sync_data, reduce_op=op)

            # 输出metrics
            train_metric_dict, train_target_metric = self._metrics.metric

            record.train_metric = train_metric_dict
            record.train_target_metric = train_target_metric
            
            if self.is_distributed:

                if self._distributed_func_wrapper is not None \
                        and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                    logging.info(f"Train epoch: {epoch}, "
                                 f"loss: {train_loss}, "
                                 f"target metric: {train_target_metric.name}:{train_target_metric.value},"
                                 f"metrics: {json2str(train_metric_dict)}")
            else:
                logging.info(f"Train epoch: {epoch}, "
                             f"loss: {train_loss}, "
                             f"target metric: {train_target_metric.name}:{train_target_metric.value},"
                             f"metrics: {json2str(train_metric_dict)}")

            self.on_train_epoch_stop(trainer=self, record=record)

            self.on_evaluate_validation_epoch_start(trainer=self, record=record)

            validation_loss = self.evaluate(validation_data_loader=validation_data_loader)

            if self.is_distributed:
                validation_loss = Sync.sync(validation_loss, device=self._device, op=ReduceOp.SUM)

            record.epoch_validation_loss = validation_loss

            if self.is_distributed:
                sync_data, op = self._metrics.to_synchronized_data()
                sync_data = Sync.sync(sync_data, device=self._device, op=op)
                self._metrics.from_synchronized_data(sync_data=sync_data, reduce_op=op)

            validation_metric_dict, validation_target_metric = self._metrics.metric

            record.validation_metric = validation_metric_dict
            record.validation_target_metric = validation_target_metric

            self._metric_tracker.add_metric(epoch=epoch,
                                            train_metric=train_metric_dict,
                                            train_model_target_metric=train_target_metric,
                                            validation_metric=validation_metric_dict,
                                            validation_model_target_metric=validation_target_metric)

            if self.is_distributed:

                if self._distributed_func_wrapper is not None \
                        and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                    logging.info(f"Evaluate Valid epoch: {epoch}, loss: {validation_loss}, "
                                 f"target metric: {validation_target_metric.name}:{validation_target_metric.value} "
                                 f"metrics: {json2str(validation_metric_dict)}")
            else:
                logging.info(f"Evaluate Valid epoch: {epoch}, loss: {validation_loss}, "
                             f"target metric: {validation_target_metric.name}:{validation_target_metric.value} "
                             f"metrics: {json2str(validation_metric_dict)}")

            self.on_evaluate_validation_epoch_stop(trainer=self, record=record)

            # 设置 lr scheduler
            # 注意这样设置会有点问题，对于某些 scheduler step 需要参数, 例如: ReduceLROnPlateau
            # 这种情况，暂时不处理, 如果处理需要重新重构，或者简单实用 isinstance ReduceLROnPlateau 来处理
            # 这里简单实用 isinstance 处理。必须指出，ReduceLROnPlateau 的基类是 object, 这也是多少有些问题。

            if self._lr_scheduler is not None:
                if isinstance(self._lr_scheduler, ReduceLROnPlateau):
                    self._lr_scheduler.step(metrics=validation_loss, epoch=epoch)
                else:
                    self._lr_scheduler.step(epoch=epoch)

            self._distributed_func_wrapper(self.save_checkpoint, epoch=epoch)

            if self._metric_tracker.early_stopping(epoch):
                if self.is_distributed:

                    if self._distributed_func_wrapper is not None \
                            and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                        logging.info(f"Epoch: {epoch}, early stopping!")
                else:
                    logging.info(f"Epoch: {epoch}, early stopping!")
                break

        self.on_training_complete(trainer=self, record=record)

    def on_train_epoch_start(self, trainer: "Trainer", record: Record) -> None:

        if self.is_distributed:

            if self._distributed_func_wrapper is not None \
                    and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                logging.info(f"on_train_epoch_start: {record.epoch}")
        else:
            logging.info(f"on_train_epoch_start: {record.epoch}")

        if self._trainer_callback is not None:
            self._trainer_callback.on_train_epoch_start(trainer=trainer,
                                                        record=record)

    def on_train_epoch_stop(self, trainer: "Trainer", record: Record) -> None:

        if self.is_distributed:

            if self._distributed_func_wrapper is not None \
                    and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                logging.info(f"on_train_epoch_stop: {record.epoch}")
        else:
            logging.info(f"on_train_epoch_stop: {record.epoch}")

        if self._trainer_callback is not None:
            self._trainer_callback.on_train_epoch_stop(trainer=trainer,
                                                       record=record)

    def on_evaluate_validation_epoch_start(self, trainer: "Trainer", record: Record) -> None:

        if self.is_distributed:

            if self._distributed_func_wrapper is not None \
                    and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                logging.info(f"on_evaluate_epoch_start: {record.epoch}")
        else:
            logging.info(f"on_evaluate_epoch_start: {record.epoch}")

        if self._trainer_callback is not None:
            self._trainer_callback.on_evaluate_validation_epoch_start(trainer=trainer,
                                                                      record=record)

    def on_evaluate_validation_epoch_stop(self, trainer: "Trainer", record: Record) -> None:

        if self.is_distributed:

            if self._distributed_func_wrapper is not None \
                    and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                logging.info(f"on_evaluate_epoch_stop: {record.epoch}")
        else:
            logging.info(f"on_evaluate_epoch_stop: {record.epoch}")

        if self._trainer_callback is not None:
            self._trainer_callback.on_evaluate_validation_epoch_stop(trainer=trainer,
                                                                     record=record)

    def on_training_complete(self, trainer: "Trainer", record: Record) -> None:

        if self.is_distributed:

            if self._distributed_func_wrapper is not None \
                    and self._distributed_func_wrapper.dst_rank == TorchDist.get_rank():
                logging.info(f"on_training_complete: {record.epoch}")
        else:
            logging.info(f"on_training_complete: {record.epoch}")

        if self._trainer_callback is not None:
            self._trainer_callback.on_training_complete(trainer=trainer,
                                                        record=record)
