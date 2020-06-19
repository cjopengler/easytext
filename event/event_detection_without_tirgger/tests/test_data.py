#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
测试 ace dataset

Authors: panxu
Date:    2020/06/09 10:12:00
"""

import os
import logging
from typing import List
from pytest import fixture
from torch.utils.data import DataLoader

from easytext.utils import bio
from easytext.utils.json_util import json2str
from easytext.utils import log_util
from easytext.data import Vocabulary, LabelVocabulary
from easytext.model import ModelInputs

from event import ROOT_PATH
from event.event_detection_without_tirgger.tests import ASSERT
from event.event_detection_without_tirgger.data import ACEDataset
from event.event_detection_without_tirgger.data import EventVocabularyCollate
from event.event_detection_without_tirgger.data import EventDataset
from event.event_detection_without_tirgger.data import EventCollate


log_util.config()


@fixture(scope="class")
def ace_dataset():
    training_data_file_path = "data/event/event_detection_without_tirgger/tests/training_data_sample.txt"
    training_data_file_path = os.path.join(ROOT_PATH, training_data_file_path)
    dataset = ACEDataset(dataset_file_path=training_data_file_path)
    return dataset


@fixture(scope="class")
def vocabulary(ace_dataset):
    vocab_collate_fn = EventVocabularyCollate()
    data_loader = DataLoader(ace_dataset, collate_fn=vocab_collate_fn)

    event_types_list: List[List[str]] = list()
    tokens_list: List[List[str]] = list()
    entity_tags_list: List[List[str]] = list()

    for collate_dict in data_loader:
        event_types_list.extend(collate_dict["event_types"])
        tokens_list.extend(collate_dict["tokens"])
        entity_tags_list.extend(collate_dict["entity_tags"])

    negative_event_type = "Negative"
    event_type_vocab = Vocabulary(tokens=event_types_list,
                                  unk=negative_event_type,
                                  padding="",
                                  special_first=True)

    word_vocab = Vocabulary(tokens=tokens_list,
                            unk=Vocabulary.UNK,
                            padding=Vocabulary.PADDING,
                            special_first=True)

    entity_tag_vocab = LabelVocabulary(entity_tags_list, padding=LabelVocabulary.PADDING)
    return {"event_type_vocab": event_type_vocab,
            "word_vocab": word_vocab,
            "entity_tag_vocab": entity_tag_vocab}


@fixture(scope="class")
def event_dataset(vocabulary):
    event_type_vocabulary = vocabulary["event_type_vocab"]
    training_data_file_path = "data/event/event_detection_without_tirgger/tests/training_data_sample.txt"
    training_data_file_path = os.path.join(ROOT_PATH, training_data_file_path)

    dataset = EventDataset(dataset_file_path=training_data_file_path,
                           event_type_vocabulary=event_type_vocabulary)
    return dataset


def test_ace_dataset(ace_dataset):
    """
    测试 ace dataset
    """

    ASSERT.assertEqual(len(ace_dataset), 3)

    expect_tokens = ["even", "as", "the", "secretary", "of", "homeland", "security", "was", "putting"]
    instance_0 = ace_dataset[0]

    instance_0_tokens = [t.text for t in instance_0["sentence"]][0:len(expect_tokens)]
    ASSERT.assertListEqual(expect_tokens, instance_0_tokens)

    expect_event_types = {"Movement:Transport"}
    instance_0_event_types = set(instance_0["event_types"])
    ASSERT.assertSetEqual(expect_event_types, instance_0_event_types)

    expect_tags = [
        {
            "text": "Secretary",
            "entity-type": "PER:Individual",
            "head": {
                "text": "Secretary",
                "start": 38,
                "end": 39
            },
            "entity_id": "CNN_CF_20030303.1900.00-E1-2",
            "start": 38,
            "end": 39
        },
        {
            "text": "the secretary of homeland security",
            "entity-type": "PER:Individual",
            "head": {
                "text": "secretary",
                "start": 3,
                "end": 4
            },
            "entity_id": "CNN_CF_20030303.1900.00-E1-188",
            "start": 2,
            "end": 7
        },
        {
            "text": "his",
            "entity-type": "PER:Individual",
            "head": {
                "text": "his",
                "start": 9,
                "end": 10
            },
            "entity_id": "CNN_CF_20030303.1900.00-E1-190",
            "start": 9,
            "end": 10
        },
        {
            "text": "Secretary Ridge",
            "entity-type": "PER:Individual",
            "head": {
                "text": "Ridge",
                "start": 39,
                "end": 40
            },
            "entity_id": "CNN_CF_20030303.1900.00-E1-198",
            "start": 38,
            "end": 40
        },
        {
            "text": "American",
            "entity-type": "GPE:Nation",
            "head": {
                "text": "American",
                "start": 29,
                "end": 30
            },
            "entity_id": "CNN_CF_20030303.1900.00-E3-196",
            "start": 29,
            "end": 30
        },
        {
            "text": "homeland security",
            "entity-type": "ORG:Government",
            "head": {
                "text": "homeland security",
                "start": 5,
                "end": 7
            },
            "entity_id": "CNN_CF_20030303.1900.00-E55-162",
            "start": 5,
            "end": 7
        },
        {
            "text": "his people",
            "entity-type": "PER:Group",
            "head": {
                "text": "people",
                "start": 10,
                "end": 11
            },
            "entity_id": "CNN_CF_20030303.1900.00-E88-171",
            "start": 9,
            "end": 11
        },
        {
            "text": "a 30-foot Cuban patrol boat with four heavily armed men",
            "entity-type": "VEH:Water",
            "head": {
                "text": "boat",
                "start": 21,
                "end": 22
            },
            "entity_id": "CNN_CF_20030303.1900.00-E96-192",
            "start": 17,
            "end": 27
        },
        {
            "text": "Cuban",
            "entity-type": "GPE:Nation",
            "head": {
                "text": "Cuban",
                "start": 19,
                "end": 20
            },
            "entity_id": "CNN_CF_20030303.1900.00-E97-193",
            "start": 19,
            "end": 20
        },
        {
            "text": "four heavily armed men",
            "entity-type": "PER:Group",
            "head": {
                "text": "men",
                "start": 26,
                "end": 27
            },
            "entity_id": "CNN_CF_20030303.1900.00-E98-194",
            "start": 23,
            "end": 27
        },
        {
            "text": "American shores",
            "entity-type": "LOC:Region-General",
            "head": {
                "text": "shores",
                "start": 30,
                "end": 31
            },
            "entity_id": "CNN_CF_20030303.1900.00-E99-195",
            "start": 29,
            "end": 31
        },
        {
            "text": "the Coast Guard",
            "entity-type": "ORG:Government",
            "head": {
                "text": "Coast Guard",
                "start": 36,
                "end": 38
            },
            "entity_id": "CNN_CF_20030303.1900.00-E102-197",
            "start": 35,
            "end": 38
        },
        {
            "text": "last month",
            "entity-type": "TIM:time",
            "head": {
                "text": "last month",
                "start": 14,
                "end": 16
            },
            "entity_id": "CNN_CF_20030303.1900.00-T4-1",
            "start": 14,
            "end": 16
        },
        {
            "text": "now",
            "entity-type": "TIM:time",
            "head": {
                "text": "now",
                "start": 40,
                "end": 41
            },
            "entity_id": "CNN_CF_20030303.1900.00-T5-1",
            "start": 40,
            "end": 41
        }
    ]

    expect_tags = {(tag["entity-type"], tag["head"]["start"], tag["head"]["end"]) for tag in expect_tags}
    instance_0_entity_tag = [t for t in instance_0["entity_tag"]]
    spans = bio.decode_one_sequence_label_to_span(instance_0_entity_tag)

    tags = {(span["label"], span["begin"], span["end"]) for span in spans}

    ASSERT.assertSetEqual(expect_tags, tags)


def test_event_vocabulary_collate(vocabulary):
    """
    测试 event vocabulary collate
    """
    event_type_vocab, word_vocab = vocabulary["event_type_vocab"], vocabulary["word_vocab"]

    negative_event_type = "Negative"
    expect_event_types = ["Movement:Transport", "Personnel:Elect", negative_event_type]

    ASSERT.assertEqual(len(expect_event_types), event_type_vocab.size)

    # 粗糙的测试，size 应该是比10大，具体的没有去文件中数一数
    ASSERT.assertTrue(word_vocab.size > 10)


def test_event_dataset(event_dataset, vocabulary):
    """
    测试 event dataset
    """

    event_type_vocab, word_vocab = vocabulary["event_type_vocab"], vocabulary["word_vocab"]

    expect_tokens = ["even", "as", "the", "secretary", "of", "homeland", "security", "was", "putting"]

    ASSERT.assertEqual(3 * event_type_vocab.size, len(event_dataset))

    hit = False
    for instance in event_dataset:
        if instance["metadata"]["sentence"][0:len("Even as the")] == "Even as the":
            hit = True
            if instance["event_type"] == "Movement:Transport":
                ASSERT.assertEqual(instance["label"], 1)
            else:
                ASSERT.assertEqual(instance["label"], 0)

            tokens = [t.text for t in instance["sentence"]][0:len(expect_tokens)]
            ASSERT.assertListEqual(expect_tokens, tokens)

    ASSERT.assertTrue(hit)


def test_event_collate(event_dataset, vocabulary):
    """
    测试 event collate
    """
    event_type_vocab = vocabulary["event_type_vocab"]
    word_vocab = vocabulary["word_vocab"]
    entity_tag_vocab = vocabulary["entity_tag_vocab"]

    event_collate = EventCollate(word_vocabulary=word_vocab,
                                 event_type_vocabulary=event_type_vocab,
                                 entity_tag_vocabulary=entity_tag_vocab,
                                 sentence_max_len=512)

    data_loader = DataLoader(dataset=event_dataset, collate_fn=event_collate, batch_size=1)

    for i, batch_instance in enumerate(data_loader):
        batch_instance: ModelInputs = batch_instance
        sentence_input = batch_instance.model_inputs["sentence"]
        ASSERT.assertEqual(1, batch_instance.batch_size)
        ASSERT.assertEqual(1, sentence_input.size(0))

        ASSERT.assertEqual(2, sentence_input.dim())

        # 粗略的计算长度
        ASSERT.assertTrue(sentence_input.size(-1) > 5)

        logging.debug(f"Instance: {i}: \n {json2str(batch_instance)}")





