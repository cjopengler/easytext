#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
ner launcher

Authors: PanXu
Date:    2020/12/21 22:09:00
"""
from typing import Optional, Union, List

import torch
import os
import logging
from typing import Dict, Union
import shutil

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer

from easytext.trainer import Trainer
from easytext.data import Vocabulary, LabelVocabulary, PretrainedVocabulary
from easytext.data import GloveLoader, SGNSLoader
from easytext.utils import log_util
from easytext.trainer import Launcher, DistributedParameter, Config

from ner.models import RnnWithCrf, BertWithCrf

from ner.data.dataset import Conll2003Dataset, MsraDataset
from ner.data import VocabularyCollate
from ner.data import NerModelCollate, BertModelCollate
from ner.loss import NerLoss
from ner.loss import NerLoss
from ner.metrics import NerModelMetricAdapter
from ner.optimizer import NerOptimizerFactory, BertOptimizerFactory
from ner.label_decoder import NerMaxModelLabelDecoder
from ner.label_decoder import NerCRFModelLabelDecoder
from ner import ROOT_PATH


class NerLauncher(Launcher):
    """
    ner 训练的启动器
    """

    NEW_TRAIN = 0

    def __init__(self, config_file_path: str, train_type: int):
        config = Config(config_file_path=config_file_path, is_training=True)
        super().__init__(config)
        self._train_type = train_type

    def _init_devices(self) -> Union[List[torch.device]]:
        return [torch.device(device) for device in self.config.devices]

    def _init_distributed_parameter(self) -> Optional[DistributedParameter]:
        if len(self._devices) > 1:
            return self.config.distributed_parameter
        else:
            return None

    def _preprocess(self):
        serialize_dir = self.config.serialize_dir
        if self._train_type == NerLauncher.NEW_TRAIN:
            # 清理 serialize dir
            if os.path.isdir(serialize_dir):
                shutil.rmtree(serialize_dir)
                os.makedirs(serialize_dir)

    def _start(self, rank: Optional[int], device: torch.device) -> None:

        is_distributed = rank is not None

        trainer = Trainer(serialize_dir=self.config.serialize_dir,
                          num_epoch=self.config.num_epoch,
                          model=self.config.model,
                          loss=self.config.loss,
                          metrics=self.config.metric,
                          optimizer_factory=self.config.optimizer,
                          lr_scheduler_factory=None,
                          patient=self.config.patient,
                          num_check_point_keep=self.config.num_check_point_keep,
                          device=device,
                          is_distributed=is_distributed)

        train_sampler = None

        if is_distributed:
            train_sampler = DistributedSampler(dataset=self.config.training_dataset, shuffle=True)

        train_data_loader = DataLoader(
            dataset=self.config.training_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=(train_sampler is None),
            num_workers=0,
            collate_fn=self.config.model_collate,
            sampler=train_sampler
        )

        validation_sampler = None

        if is_distributed:
            validation_sampler = DistributedSampler(dataset=self.config.training_dataset, shuffle=False)

        validation_data_loader = DataLoader(
            dataset=self.config.validation_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.config.model_collate,
            sampler=validation_sampler
        )

        trainer.train(train_data_loader=train_data_loader,
                      validation_data_loader=validation_data_loader)


if __name__ == '__main__':
    log_util.config(level=logging.INFO)

    config_file_path = "data/ner/rnn_with_crf/config.json"
    config_file_path = "data/ner/bert_with_crf/config_cpu.json"
    config_file_path = "data/ner/bert_with_crf/config_multi_gpu.json"

    config_file_path = os.path.join(ROOT_PATH, config_file_path)

    ner_launcher = NerLauncher(config_file_path=config_file_path, train_type=NerLauncher.NEW_TRAIN)
    ner_launcher()
