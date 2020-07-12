#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
生成训练样本数据集

Authors: PanXu
Date:    2020/07/12 15:40:00
"""
from typing import List
from torch.utils.data import Dataset

from easytext.data import Instance

from .sem_eval_dataset import SemEvalDataset


class ACSASemEvalDataset(Dataset):
    """
    将 SemEval 数据集 构造成 ACSA 用的格式。在 SemEval 中, 格式:
    sentence 对应多个 category 和 polarity。所以，需要拆分开，
    形成 <sentence, category, polarity> 作为一个样本。

    """

    def __init__(self, dataset_file_path: str):
        self._sem_eval_dataset = SemEvalDataset(dataset_file_path=dataset_file_path)

        self._instances: List[Instance] = list()

        for sem_eval_instance in self._sem_eval_dataset:

            sentence = sem_eval_instance["sentence"]
            aspect_categories = sem_eval_instance["aspect_categories"]

            for aspect_category in aspect_categories:
                instance = Instance()
                instance["sentence"] = sentence
                instance["category"] = aspect_category["category"]
                instance["label"] = aspect_category["polarity"]

                self._instances.append(instance)

    def __getitem__(self, index: int) -> Instance:
        return self._instances[index]

    def __len__(self) -> int:
        return len(self._instances)
