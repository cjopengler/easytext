#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
config factory

Authors: PanXu
Date:    2020/09/10 21:34:00
"""
import os
from typing import Dict
import shutil

import torch


from ner import ROOT_PATH
from ner.models import NerV1, NerV2, NerV3, NerV4

from ner.data.dataset import Conll2003Dataset, MsraDataset
from ner.config.conll2003_dataset_config import Conll2003DatasetConfig
from ner.config.msra_dataset_config import MsraDatasetConfig


class NerConfigFactory:
    """
    Ner Config Factory 子类
    """

    def __init__(self, model_name: str, dataset_name: str, debug: bool):
        """
        初始化
        :param model_name: 模型名字, 不同的模型名字配置不同
        :param dataset_name: 数据集名字
        :param debug: True: debug 模型; False: 非debug模型
        """
        self.debug = debug
        self.model_name = model_name
        self.dataset_name = dataset_name

    def create(self) -> Dict:
        """
        创建 config
        :return: config 字典
        """

        config = dict()

        if self.dataset_name == Conll2003Dataset.NAME:
            dataset_config = Conll2003DatasetConfig(debug=self.debug)
        elif self.dataset_name == MsraDataset.NAME:
            dataset_config = MsraDatasetConfig(debug=self.debug)
        else:
            raise RuntimeError(f"dataset name: {self.dataset_name} 是非法的!")

        config["dataset_name"] = self.dataset_name

        config["train_dataset_file_path"] = dataset_config.train_dataset_file_path

        config["validation_dataset_file_path"] = dataset_config.validation_dataset_file_path

        serialize_dir = f"data/ner/{self.dataset_name}_{self.model_name}/train"
        serialize_dir = os.path.join(ROOT_PATH, serialize_dir)

        if os.path.isdir(serialize_dir):
            shutil.rmtree(serialize_dir)

        os.makedirs(serialize_dir)

        config["serialize_dir"] = serialize_dir
        config["patient"] = 20
        config["num_check_point_keep"] = 1

        config["num_epoch"] = 200
        config["train_batch_size"] = 12
        config["test_batch_size"] = 16
        config["model_name"] = self.model_name
        config["fine_tuning"] = True
        config["label_vocabulary_dir"] = f"data/ner/{self.dataset_name}_{self.model_name}/vocabulary/label_vocabulary"
        config["token_vocabulary_dir"] = f"data/ner/{self.dataset_name}_{self.model_name}/vocabulary/token_vocabulary"

        pretrained_word_embedding_file_path = "data/pretrained/glove/glove.6B.100d.txt"

        config["pretrained_word_embedding_file_path"] = os.path.join(ROOT_PATH,
                                                                     pretrained_word_embedding_file_path)

        bert_dir = os.path.join(ROOT_PATH,
                                "data/pretrained/bert/bert-base-chinese-pytorch")

        if self.model_name == NerV1.NAME:
            model_config = {
                "word_embedding_dim": 100,
                "hidden_size": 200,
                "num_layer": 2,
                "dropout": 0.4
            }
        elif self.model_name == NerV2.NAME:
            model_config = {
                "word_embedding_dim": 100,  # glove 6B 100d
                "hidden_size": 200,
                "num_layer": 2,
                "dropout": 0.4
            }
        elif self.model_name == NerV3.NAME:
            model_config = {
                "word_embedding_dim": 100,  # glove 6B 100d
                "hidden_size": 200,
                "num_layer": 2,
                "dropout": 0.4
            }
        elif self.model_name == NerV4.NAME:
            config["bert_dir"] = bert_dir

            model_config = {
                "bert_dir": bert_dir,
                "dropout": 0.1,   # 参考 bert config
                "is_used_crf": False  # 不使用 crf
            }
        else:
            raise RuntimeError(f"无效的 model name: {self.model_name}")

        config["model"] = model_config

        if torch.cuda.is_available():
            config["cuda"] = ["cuda:0"]
        else:
            config["cuda"] = None

        return config

