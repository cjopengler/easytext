#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
event 样本处理 包括 padding 以及 truncate, indexing 等处理，转化成模型需要输入

Authors: panxu(panxu@baidu.com)
Date:    2020/06/03 17:40:00
"""
import torch
from typing import Iterable, List

from easytext.data import Instance
from easytext.data import PretrainedVocabulary, Vocabulary
from easytext.data import LabelVocabulary
from easytext.data.model_collate import ModelCollate, ModelInputs


class EventCollate(ModelCollate):
    """
    对事件的数据进行 index padding等操作，生成训练样本，对应的是 event_dataset.EventDataset
    """

    def __init__(self,
                 word_vocabulary: PretrainedVocabulary,
                 event_type_vocabulary: Vocabulary,
                 entity_tag_vocabulary: LabelVocabulary,
                 sentence_max_len=512):
        super().__init__()
        self._word_vocab = word_vocabulary
        self._event_type_vocab = event_type_vocabulary
        self._entity_tag_vocab = entity_tag_vocabulary
        self._max_sentence_len = sentence_max_len

    def __call__(self, instances: Iterable[Instance]) -> ModelInputs:

        if not isinstance(instances, list):
            raise RuntimeError("instances 应该是 list")

        batch_max_sentence_len: int = 0

        for instance in instances:
            sentence = instance["sentence"]

            if len(sentence) > batch_max_sentence_len:
                batch_max_sentence_len = len(sentence)

        if batch_max_sentence_len > self._max_sentence_len:
            batch_max_sentence_len = self._max_sentence_len

        token_indices_list = list()
        event_type_indices = list()
        entity_tag_indices_list = list()
        metadatas = list()
        labels: List[int] = None

        for instance in instances:

            token_indices = [self._word_vocab.index(self._word_vocab.padding)] * batch_max_sentence_len
            token_indices_list.append(token_indices)

            for i, token in enumerate(instance["sentence"][0:batch_max_sentence_len]):
                token_indices[i] = self._word_vocab.index(token.text)

            event_type_index = self._event_type_vocab.index(instance["event_type"])
            event_type_indices.append(event_type_index)

            entity_tag_indices = [self._entity_tag_vocab.index(self._entity_tag_vocab.padding)] * batch_max_sentence_len
            entity_tag_indices_list.append(entity_tag_indices)

            for i, entity_tag in enumerate(instance["entity_tag"][0:batch_max_sentence_len]):
                entity_tag_indices[i] = self._entity_tag_vocab.index(entity_tag)

            metadatas.append(instance["metadata"])
            # label 要单独处理，因为在 predict 的时候，是没有的
            if "label" in instance:

                if labels is None:
                    labels = list()

                labels.append(instance["label"])

        token_indices_tensor = torch.tensor(token_indices_list, dtype=torch.long)
        event_type_tensor = torch.tensor(event_type_indices, dtype=torch.long)
        entity_tag_indices_tensor = torch.tensor(entity_tag_indices_list, dtype=torch.long)
        label_tensor = torch.tensor(labels, dtype=torch.long)

        batch_size = len(instances)
        model_inputs = {"sentence": token_indices_tensor,
                        "entity_tag": entity_tag_indices_tensor,
                        "event_type": event_type_tensor,
                        "metadata": metadatas}

        model_inputs = ModelInputs(batch_size=batch_size,
                                   model_inputs=model_inputs,
                                   labels=label_tensor)
        return model_inputs
