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

import torch

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from easytext.trainer import ConfigFactory
from easytext.trainer import Trainer
from easytext.data import Vocabulary, LabelVocabulary, PretrainedVocabulary
from easytext.data import GloveLoader
from easytext.utils import log_util

from ner import ROOT_PATH
from ner.models import NerV1, NerV2, NerV3
from ner.data.dataset import Conll2003Dataset
from ner.data import VocabularyCollate
from ner.data import NerModelCollate
from ner.loss import NerLoss
from ner.loss import NerCRFLoss
from ner.metrics import NerModelMetricAdapter
from ner.optimizer import NerOptimizerFactory
from ner.label_decoder import NerMaxModelLabelDecoder
from ner.label_decoder import NerCRFModelLabelDecoder


class NerConfigFactory(ConfigFactory):
    """
    Ner Config Factory 子类
    """

    def __init__(self, model_name: str, debug: bool = True):
        """
        初始化
        :param model_name: 模型名字, 不同的模型名字配置不同
        :param debug: True: debug 模型; False: 非debug模型
        """
        self.debug = debug
        self.model_name = model_name

    def create(self) -> Dict:
        """
        创建 config
        :return: config 字典
        """

        config = dict()

        if self.debug:
            train_dataset_file_path = "data/conll2003/sample.txt"
        else:
            train_dataset_file_path = "data/conll2003/eng.train"
        train_dataset_file_path = os.path.join(ROOT_PATH, train_dataset_file_path)

        config["train_dataset_file_path"] = train_dataset_file_path

        if self.debug:
            validation_dataset_file_path = "data/conll2003/sample.txt"
        else:
            validation_dataset_file_path = "data/conll2003/eng.testa"
        validation_dataset_file_path = os.path.join(ROOT_PATH, validation_dataset_file_path)

        config["validation_dataset_file_path"] = validation_dataset_file_path

        serialize_dir = f"data/ner/conll2003_{self.model_name}/train"
        serialize_dir = os.path.join(ROOT_PATH, serialize_dir)
        config["serialize_dir"] = serialize_dir
        config["patient"] = 20
        config["num_check_point_keep"] = 10

        config["num_epoch"] = 100
        config["model_name"] = self.model_name

        pretrained_word_embedding_file_path = "data/pretrained/glove/glove.6B.100d.txt"

        config["pretrained_word_embedding_file_path"] = os.path.join(ROOT_PATH,
                                                                     pretrained_word_embedding_file_path)

        if self.model_name == ModelName.NER_V1:
            model_config = {
                "word_embedding_dim": 100,
                "hidden_size": 200,
                "num_layer": 2,
                "dropout": 0.4
            }
        elif self.model_name == ModelName.NER_V2:
            model_config = {
                "word_embedding_dim": 100,  # glove 6B 100d
                "hidden_size": 200,
                "num_layer": 2,
                "dropout": 0.4
            }
        elif self.model_name == ModelName.NER_V3:
            model_config = {
                "word_embedding_dim": 100,  # glove 6B 100d
                "hidden_size": 200,
                "num_layer": 2,
                "dropout": 0.4
            }
        else:
            raise RuntimeError(f"无效的 model name: {self.model_name}")

        config["model"] = model_config

        if torch.cuda.is_available():
            config["cuda"] = ["cuda:0"]
        else:
            config["cuda"] = None

        return config


class Train:
    NEW_TRAIN = 0  # 全新训练
    RECOVERY_TRAIN = 1  # 恢复训练

    def __init__(self, train_type: int, config: Dict):
        """
        初始化
        :param train_model: 训练类型 NEW_TRAIN: 全新训练, RECOVERY_TRAIN: 恢复训练
        :param config: 配置参数
        """
        self._train_type = train_type
        self.config = config

    def build_vocabulary(self, dataset: Dataset):

        data_loader = DataLoader(dataset=dataset,
                                 batch_size=100,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=VocabularyCollate())
        batch_tokens = list()
        batch_sequence_labels = list()

        for collate_dict in data_loader:
            batch_tokens.extend(collate_dict["tokens"])
            batch_sequence_labels.extend(collate_dict["sequence_labels"])

        token_vocabulary = Vocabulary(tokens=batch_tokens,
                                      padding=Vocabulary.PADDING,
                                      unk=Vocabulary.UNK,
                                      special_first=True)
        model_name = self.config["model_name"]

        if model_name in {ModelName.NER_V2, ModelName.NER_V3}:
            pretrained_word_embedding_file_path = self.config["pretrained_word_embedding_file_path"]
            glove_loader = GloveLoader(embedding_dim=100,
                                       pretrained_file_path=pretrained_word_embedding_file_path)

            token_vocabulary = PretrainedVocabulary(vocabulary=token_vocabulary,
                                                    pretrained_word_embedding_loader=glove_loader)

        label_vocabulary = LabelVocabulary(labels=batch_sequence_labels,
                                           padding=LabelVocabulary.PADDING)

        return {"token_vocabulary": token_vocabulary,
                "label_vocabulary": label_vocabulary}

    def build_model(self,
                    token_vocabulary: Union[Vocabulary, PretrainedVocabulary],
                    label_vocabulary: LabelVocabulary):
        model_name = config["model_name"]

        model_config = config["model"]

        if model_name == ModelName.NER_V1:
            model = NerV1(token_vocabulary=token_vocabulary,
                          label_vocabulary=label_vocabulary,
                          word_embedding_dim=model_config["word_embedding_dim"],
                          hidden_size=model_config["hidden_size"],
                          num_layer=model_config["num_layer"],
                          dropout=model_config["dropout"])
        elif model_name == ModelName.NER_V2:
            model = NerV2(token_vocabulary=token_vocabulary,
                          label_vocabulary=label_vocabulary,
                          word_embedding_dim=model_config["word_embedding_dim"],
                          hidden_size=model_config["hidden_size"],
                          num_layer=model_config["num_layer"],
                          dropout=model_config["dropout"])
        elif model_name == ModelName.NER_V3:
            model = NerV3(token_vocabulary=token_vocabulary,
                          label_vocabulary=label_vocabulary,
                          word_embedding_dim=model_config["word_embedding_dim"],
                          hidden_size=model_config["hidden_size"],
                          num_layer=model_config["num_layer"],
                          dropout=model_config["dropout"])

        return model

    def build_loss(self):
        model_name = config["model_name"]

        if model_name == ModelName.NER_V1:
            loss = NerLoss()
        elif model_name == ModelName.NER_V2:
            loss = NerLoss()
        elif model_name == ModelName.NER_V3:
            loss = NerCRFLoss()

        return loss

    def build_model_metric(self, label_vocabulary: LabelVocabulary):
        model_name = config["model_name"]

        if model_name == ModelName.NER_V1:
            metric = NerModelMetricAdapter(label_vocabulary=label_vocabulary,
                                           model_label_decoder=NerMaxModelLabelDecoder(
                                               label_vocabulary=label_vocabulary))
        elif model_name == ModelName.NER_V2:
            metric = NerModelMetricAdapter(label_vocabulary=label_vocabulary,
                                           model_label_decoder=NerMaxModelLabelDecoder(
                                               label_vocabulary=label_vocabulary))
        elif model_name == ModelName.NER_V3:
            metric = NerModelMetricAdapter(label_vocabulary=label_vocabulary,
                                           model_label_decoder=NerCRFModelLabelDecoder(
                                               label_vocabulary=label_vocabulary))

        return metric

    def __call__(self):
        config = self.config
        serialize_dir = config["serialize_dir"]

        if self._train_type == Train.NEW_TRAIN:
            # 清理 serialize dir
            if os.path.isdir(serialize_dir):
                shutil.rmtree(serialize_dir)
                os.makedirs(serialize_dir)

        num_epoch = config["num_epoch"]
        patient = config["patient"]
        num_check_point_keep = config["num_check_point_keep"]

        train_dataset = Conll2003Dataset(dataset_file_path=config["train_dataset_file_path"])
        validation_dataset = Conll2003Dataset(dataset_file_path=config["validation_dataset_file_path"])

        # 构建 vocabulary
        vocab_dict = self.build_vocabulary(dataset=train_dataset)
        token_vocabulary = vocab_dict["token_vocabulary"]
        label_vocabulary = vocab_dict["label_vocabulary"]

        model = self.build_model(token_vocabulary=token_vocabulary,
                                 label_vocabulary=label_vocabulary)

        loss = self.build_loss()

        metric = self.build_model_metric(label_vocabulary=label_vocabulary)

        cuda = config["cuda"]

        trainer = Trainer(serialize_dir=serialize_dir,
                          num_epoch=num_epoch,
                          model=model,
                          loss=loss,
                          metrics=metric,
                          optimizer_factory=NerOptimizerFactory(),
                          lr_scheduler_factory=None,
                          patient=patient,
                          num_check_point_keep=num_check_point_keep,
                          cuda_devices=cuda)

        model_collate = NerModelCollate(
            token_vocab=token_vocabulary,
            sequence_label_vocab=label_vocabulary,
            sequence_max_len=512
        )

        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=0,
            collate_fn=model_collate
        )

        validation_data_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0,
            collate_fn=model_collate
        )

        trainer.train(train_data_loader=train_data_loader,
                      validation_data_loader=validation_data_loader)


class ModelName:
    """
    模型名字定义
    """

    NER_V1 = "ner_v1"  # bilstm
    NER_V2 = "ner_v2"  # bilstm + glove.6b.100d
    NER_V3 = "ner_v3"  # bilstm + glove.6b.100d + crf


if __name__ == '__main__':
    log_util.config(level=logging.INFO)

    config = NerConfigFactory(debug=True, model_name=ModelName.NER_V3).create()
    Train(train_type=Train.NEW_TRAIN, config=config)()
