#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
将 ACEDataset 转换成 event 样本。因为在 ACEDataset 中返回的每一个 instance 是有多个事件类型，
需要根据现在的场景替换成，将 event_type 与 每一个 instance 转换成一条样本。

因为在不同的模型下处理的方式是不同的，这个类仅仅是用来处理 "event_detection_without_trigger" 算法

Authors: panxu
Date:    2020/06/07 18:01:00
"""
from typing import List
from torch.utils.data import Dataset

from easytext.data import Instance
from easytext.data import Vocabulary

from event.event_detection_without_tirgger.data.ace_dataset import ACEDataset


class EventDataset(Dataset):
    """
    ACE event dataset
    """

    def __init__(self, dataset_file_path: str, event_type_vocabulary: Vocabulary):
        """
        初始化 ACE Event Dataset
        :param dataset_file_path: 数据集的文件路基
        """
        super().__init__()
        self._ace_dataset = ACEDataset(dataset_file_path=dataset_file_path)

        self._instances: List[Instance] = list()

        for ori_instance in self._ace_dataset:

            ori_event_types = ori_instance["event_types"]

            ori_event_type_set = None

            if ori_event_types is not None:  # 实际预测的时候 ori_event_types is None
                # 针对 training 和 validation 设置，因为 对于 pair<sentence, unk>, label = 1
                ori_event_type_set = set(ori_event_types)

                if len(ori_event_type_set) == 0:
                    ori_event_type_set.add(event_type_vocabulary.unk)

            for index in range(event_type_vocabulary.size):
                # 遍历所有的label, 形成 pair<句子,事件类型>，作为样本
                event_type = event_type_vocabulary.token(index)

                instance = Instance()

                instance["sentence"] = ori_instance["sentence"]

                instance["entity_tag"] = ori_instance["entity_tag"]

                instance["event_type"] = event_type
                instance["metadata"] = ori_instance["metadata"]

                if ori_event_type_set is not None:
                    if event_type in ori_event_type_set:
                        instance["label"] = 1
                    else:
                        instance["label"] = 0
                else:
                    # 是针对实际的 prediction 设置的
                    pass

                self._instances.append(instance)

    def __getitem__(self, index: int) -> Instance:
        return self._instances[index]

    def __len__(self) -> int:
        return len(self._instances)
