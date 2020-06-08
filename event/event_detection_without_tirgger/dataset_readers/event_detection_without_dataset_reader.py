#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
doc: Event Detection without Triggers

paper: Event Detection without Triggers
Authors: panxu(panxu@baidu.com)
Date:    2020/01/28 08:08:00
"""
import json
from typing import Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import TextField, LabelField, MetadataField
from allennlp.data.tokenizers import Token
from allennlp.data.token_indexers import TokenIndexer

from zznlp.models.event_detection_without_tirgger.types_define import EVENT_TYPES
from zznlp.models.event_detection_without_tirgger.types_define import NEGATIVE_EVENT_TYPE
from zznlp.models.event_detection_without_tirgger.types_define import NEGATIVE_ENTITY_TYPE


@DatasetReader.register("EventDetectionWithoutTriggerDatasetReader")
class EventDetectionWithoutTriggerDatasetReader(DatasetReader):
    """
    Event Detection without Triggers 数据读取

    ACL 2019 reference: https://www.aclweb.org/anthology/N19-1080/
    """

    EVENT_TYPE_NAMESPACE = "event_type_tags"

    def __init__(self,
                 token_indexer: TokenIndexer,
                 entity_tag_indexer: TokenIndexer,
                 lazy: bool = False):

        super().__init__(lazy=lazy)

        self._token_indexer = token_indexer
        self._entity_tag_indexer = entity_tag_indexer

    def _read(self, file_path: str) -> Iterable[Instance]:

        predefine_event_types = EVENT_TYPES

        with open(file_path) as f:
            for line in f:
                line = line.strip()

                if line:
                    item = json.loads(line)
                    sentence = item["sentence"]
                    tokens = item["words"]
                    entity_tags = [NEGATIVE_ENTITY_TYPE for _ in tokens]

                    for entity_mention in item["golden-entity-mentions"]:
                        entity_type = entity_mention["entity-type"]
                        start = entity_mention["head"]["start"]
                        end = entity_mention["head"]["end"]
                        for i in range(start, end):
                            entity_tags[i] = entity_type

                    # 在paper中提到，当event有多个 类型 的时候，使用一个类型
                    if "golden-event-mentions" in item:
                        event_types = {event["event_type"] for event in item["golden-event-mentions"]}

                        if len(event_types) == 0:
                            event_types.add(NEGATIVE_EVENT_TYPE)

                        for predefine_event_type in predefine_event_types:

                            if predefine_event_type in event_types:
                                label = 1
                            else:
                                label = 0

                            yield self.text_to_instance(sentence=sentence,
                                                        tokens=tokens,
                                                        entity_tags=entity_tags,
                                                        event_type=predefine_event_type,
                                                        label=label)
                    else:
                        for predefine_event_type in predefine_event_types:
                            yield self.text_to_instance(sentence=sentence,
                                                        tokens=tokens,
                                                        entity_tags=entity_tags,
                                                        event_type=predefine_event_type,
                                                        label=None)

    def text_to_instance(self,
                         sentence: str,
                         tokens: List[str],
                         entity_tags: List[str],
                         event_type: str,
                         label: int = None) -> Instance:

        sentence_field = TextField(tokens=[Token(t) for t in tokens],
                                   token_indexers={"tokens": self._token_indexer})
        entity_tag_field = TextField(tokens=[Token(t) for t in entity_tags],
                                     token_indexers={"entity_tags": self._entity_tag_indexer})
        event_type_field = LabelField(label=event_type,
                                      label_namespace=EventDetectionWithoutTriggerDatasetReader.EVENT_TYPE_NAMESPACE)
        label_field = LabelField(label=label,
                                 label_namespace="labels",
                                 skip_indexing=True)
        metadata_field = MetadataField({"sentence": sentence,
                                        "event_type": event_type,
                                        "label": label})

        fields = {"sentence": sentence_field,
                  "entity_tag": entity_tag_field,
                  "event_type": event_type_field,
                  "label": label_field,
                  "metadata": metadata_field}

        return Instance(fields=fields)
