#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
训练

Authors: PanXu
Date:    2020/06/27 21:48:00
"""
import os
import logging
from typing import Dict, Union
import shutil

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import BertTokenizer

from easytext.trainer import Trainer
from easytext.data import Vocabulary, LabelVocabulary, PretrainedVocabulary
from easytext.data import GloveLoader, SGNSLoader
from easytext.utils import log_util
from easytext.trainer import Config

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


class Train:
    NEW_TRAIN = 0  # 全新训练
    RECOVERY_TRAIN = 1  # 恢复训练

    def __init__(self, train_type: int, config_file_path: str):
        """
        初始化
        :param train_model: 训练类型 NEW_TRAIN: 全新训练, RECOVERY_TRAIN: 恢复训练
        :param config_file_path: 配置文件路径
        """
        self._train_type = train_type
        is_training = self._train_type == Train.NEW_TRAIN
        self.config = Config(is_training=is_training,
                             config_file_path=config_file_path)

    def __call__(self):

        serialize_dir = self.config.serialize_dir

        if self._train_type == Train.NEW_TRAIN:
            # 清理 serialize dir
            if os.path.isdir(serialize_dir):
                shutil.rmtree(serialize_dir)
                os.makedirs(serialize_dir)

        trainer = Trainer(serialize_dir=serialize_dir,
                          num_epoch=self.config.num_epoch,
                          model=self.config.model,
                          loss=self.config.loss,
                          metrics=self.config.metric,
                          optimizer_factory=self.config.optimizer,
                          lr_scheduler_factory=None,
                          patient=self.config.patient,
                          num_check_point_keep=self.config.num_check_point_keep,
                          devices=self.config.cuda)

        train_data_loader = DataLoader(
            dataset=self.config.training_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.config.model_collate
        )

        validation_data_loader = DataLoader(
            dataset=self.config.validation_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.config.model_collate
        )

        trainer.train(train_data_loader=train_data_loader,
                      validation_data_loader=validation_data_loader)


if __name__ == '__main__':
    log_util.config(level=logging.INFO)

    config_file_path = "data/ner/rnn_with_crf/config.json"
    config_file_path = "data/ner/bert_with_crf/config.json"
    config_file_path = os.path.join(ROOT_PATH, config_file_path)

    Train(train_type=Train.NEW_TRAIN, config_file_path=config_file_path)()
