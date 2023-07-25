#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
ace 数据读取

Authors: panxu(panxu@baidu.com)
Date:    2020/06/03 17:40:00
"""
import json
from typing import List
from torch.utils.data import Dataset

from easytext.data import Instance
from easytext.data.tokenizer import Token, EnTokenizer
from easytext.utils import bio


class ACEDataset(Dataset):
    """
    ACE 事件数据集
    """

    def __init__(self, dataset_file_path: str):
        """
        ACE 事件数据集 初始化.
        :param dataset_file_path: 包含数据集的文件
        """
        super().__init__()
        self._instances: List[Instance] = list()

        with open(dataset_file_path) as f:
            for line in f:
                line = line.strip()

                if line:
                    item = json.loads(line)
                    sentence = item["sentence"]
                    tokens = item["words"]
                    entity_labels = ["O" for _ in tokens]

                    for entity_mention in item["golden-entity-mentions"]:
                        entity_type = entity_mention["entity-type"]
                        start = entity_mention["head"]["start"]
                        end = entity_mention["head"]["end"]

                        bio.fill(sequence_label=entity_labels, begin_index=start, end_index=end, tag=entity_type)

                    # 在paper中提到，当event有多个 类型 的时候，使用一个类型
                    if "golden-event-mentions" in item:
                        # 在 training 或者 validation 数据中，golden-event-mentions 是一定存在的
                        event_types = {event["event_type"] for event in item["golden-event-mentions"]}
                        event_types = list(event_types)

                        instance = self.text_to_instance(sentence=sentence,
                                                         tokens=tokens,
                                                         entity_tags=entity_labels,
                                                         event_types=event_types)
                    else:
                        # 这种情况是预测的时候使用的，因为预测的时候是没有 golden-event-mentions 这个 key 的
                        instance = ACEDataset.text_to_instance(sentence=sentence,
                                                               tokens=tokens,
                                                               entity_tags=entity_labels,
                                                               event_types=None)

                    self._instances.append(instance)

    @staticmethod
    def text_to_instance(sentence: str,
                         tokens: List[str],
                         entity_tags: List[str],
                         event_types: List[str]) -> Instance:

        instance = Instance()

        instance["sentence"] = [Token(t.lower()) for t in tokens]

        instance["entity_tag"] = entity_tags

        instance["event_types"] = event_types
        instance["metadata"] = {"sentence": sentence,
                                "event_types": event_types}

        return instance

    def __getitem__(self, index: int) -> Instance:
        return self._instances[index]

    def __len__(self) -> int:
        return len(self._instances)
