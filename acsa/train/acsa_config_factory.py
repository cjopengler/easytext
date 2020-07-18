#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
acsa config factory

Authors: PanXu
Date:    2020/07/18 18:35:00
"""
import os
from typing import Dict

from easytext.trainer import ConfigFactory

from acsa import ROOT_PATH


class ACSAConfigFactory(ConfigFactory):
    """
    config factory
    """

    def __init__(self, debug: bool = False):
        """
        初始化
        :param debug: True: debug 模型; False: 正常训练模式
        """
        self.debug = debug

    def create(self) -> Dict:
        config = dict()
        config["debug"] = self.debug

        config["model"] = dict()
        model_config = config["model"]

        if self.debug:
            training_dataset_file_path = os.path.join(ROOT_PATH,
                                                      "data/dataset/SemEval-2014-Task-4-REST/sample.xml")
            validation_dataset_file_path = os.path.join(ROOT_PATH,
                                                      "data/dataset/SemEval-2014-Task-4-REST/sample.xml")
        else:

            training_dataset_file_path = os.path.join(ROOT_PATH,
                                                      "data/dataset/SemEval-2014-Task-4-REST/"
                                                      "SemEval'14-ABSA-TrainData_v2 & AnnotationGuidelines/"
                                                      "Restaurants_Train_v2.xml")
            validation_dataset_file_path = os.path.join(ROOT_PATH,
                                                        "data/dataset/SemEval-2014-Task-4-REST/"
                                                        "ABSA_Gold_TestData/Restaurants_Test_Gold.xml")
            config["pretrained_file_path"] = "data/pretrained/glove/glove.840B.300d.txt"

        config["training_dataset_file_path"] = training_dataset_file_path
        config["validation_dataset_file_path"] = validation_dataset_file_path

        config["serialize_dir"] = os.path.join(ROOT_PATH,
                                               "data/acsa/atae_lstm/train")
        config["num_epoches"] = 100
        config["batch_size"] = 32
        config["patient"] = 20
        config["num_check_point_keep"] = 5

        model_config["token_embedding_dim"] = 300
        model_config["category_embedding_dim"] = 300

        return config

