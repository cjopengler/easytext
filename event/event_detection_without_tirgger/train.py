#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
对模型进行训练

Authors: panxu(panxu@baidu.com)
Date:    2020/06/17 18:31:00
"""

from typing import Dict, List
import os
import shutil

import torch
from torch.utils.data import DataLoader

from easytext.utils import log_util
from easytext.data import Vocabulary, LabelVocabulary, PretrainedVocabulary
from easytext.data import GloveLoader
from easytext.trainer import Trainer
from easytext.trainer import ConfigFactory

from event import ROOT_PATH
from event.event_detection_without_tirgger.data import ACEDataset
from event.event_detection_without_tirgger.data import EventDataset
from event.event_detection_without_tirgger.data import EventCollate
from event.event_detection_without_tirgger.data import EventVocabularyCollate
from event.event_detection_without_tirgger.loss import EventLoss
from event.event_detection_without_tirgger.metrics import EventF1MetricAdapter
from event.event_detection_without_tirgger.models import EventModel
from event.event_detection_without_tirgger.optimizer import EventOptimizerFactory


class Train:
    """
    训练的入口
    """

    NEW_TRAIN = 0
    RECOVERY_TRAIN = 1

    def __call__(self, config: Dict, train_type: int):
        serialize_dir = config["serialize_dir"]
        vocabulary_dir = config["vocabulary_dir"]
        pretrained_embedding_file_path = config["pretrained_embedding_file_path"]
        word_embedding_dim = config["word_embedding_dim"]
        pretrained_embedding_max_size = config["pretrained_embedding_max_size"]
        is_fine_tuning = config["fine_tuning"]

        word_vocab_dir = os.path.join(vocabulary_dir, "vocabulary", "word_vocabulary")
        event_type_vocab_dir = os.path.join(vocabulary_dir, "vocabulary", "event_type_vocabulary")
        entity_tag_vocab_dir = os.path.join(vocabulary_dir, "vocabulary", "entity_tag_vocabulary")

        if train_type == Train.NEW_TRAIN:

            if os.path.isdir(serialize_dir):
                shutil.rmtree(serialize_dir)

            os.makedirs(serialize_dir)

            if os.path.isdir(vocabulary_dir):
                shutil.rmtree(vocabulary_dir)
            os.makedirs(vocabulary_dir)
            os.makedirs(word_vocab_dir)
            os.makedirs(event_type_vocab_dir)
            os.makedirs(entity_tag_vocab_dir)

        elif train_type == Train.RECOVERY_TRAIN:
            pass
        else:
            assert False, f"train_type: {train_type} error!"

        train_dataset_file_path = config["train_dataset_file_path"]
        validation_dataset_file_path = config["validation_dataset_file_path"]

        num_epoch = config["epoch"]
        batch_size = config["batch_size"]

        if train_type == Train.NEW_TRAIN:
            # 构建词汇表
            ace_dataset = ACEDataset(train_dataset_file_path)
            vocab_data_loader = DataLoader(dataset=ace_dataset,
                                           batch_size=10,
                                           shuffle=False, num_workers=0,
                                           collate_fn=EventVocabularyCollate())

            tokens: List[List[str]] = list()
            event_types: List[List[str]] = list()
            entity_tags: List[List[str]] = list()

            for colleta_dict in vocab_data_loader:
                tokens.extend(colleta_dict["tokens"])
                event_types.extend(colleta_dict["event_types"])
                entity_tags.extend(colleta_dict["entity_tags"])

            word_vocabulary = Vocabulary(tokens=tokens,
                                         padding=Vocabulary.PADDING,
                                         unk=Vocabulary.UNK,
                                         special_first=True)

            glove_loader = GloveLoader(embedding_dim=word_embedding_dim,
                                       pretrained_file_path=pretrained_embedding_file_path,
                                       max_size=pretrained_embedding_max_size)

            pretrained_word_vocabulary = PretrainedVocabulary(vocabulary=word_vocabulary,
                                                              pretrained_word_embedding_loader=glove_loader)

            pretrained_word_vocabulary.save_to_file(word_vocab_dir)

            event_type_vocabulary = Vocabulary(tokens=event_types,
                                               padding="",
                                               unk="Negative",
                                               special_first=True)
            event_type_vocabulary.save_to_file(event_type_vocab_dir)

            entity_tag_vocabulary = LabelVocabulary(labels=entity_tags,
                                                    padding=LabelVocabulary.PADDING)
            entity_tag_vocabulary.save_to_file(entity_tag_vocab_dir)
        else:
            pretrained_word_vocabulary = PretrainedVocabulary.from_file(word_vocab_dir)
            event_type_vocabulary = Vocabulary.from_file(event_type_vocab_dir)
            entity_tag_vocabulary = Vocabulary.from_file(entity_tag_vocab_dir)

        model = EventModel(alpha=0.5,
                           activate_score=True,
                           sentence_vocab=pretrained_word_vocabulary,
                           sentence_embedding_dim=word_embedding_dim,
                           entity_tag_vocab=entity_tag_vocabulary,
                           entity_tag_embedding_dim=50,
                           event_type_vocab=event_type_vocabulary,
                           event_type_embedding_dim=300,
                           lstm_hidden_size=300,
                           lstm_encoder_num_layer=1,
                           lstm_encoder_droupout=0.4)

        trainer = Trainer(
            serialize_dir=serialize_dir,
            num_epoch=num_epoch,
            model=model,
            loss=EventLoss(),
            optimizer_factory=EventOptimizerFactory(is_fine_tuning=is_fine_tuning),
            metrics=EventF1MetricAdapter(event_type_vocabulary=event_type_vocabulary),
            patient=10,
            num_check_point_keep=5,
            cuda_devices=None
        )

        train_dataset = EventDataset(dataset_file_path=train_dataset_file_path,
                                     event_type_vocabulary=event_type_vocabulary)
        validation_dataset = EventDataset(dataset_file_path=validation_dataset_file_path,
                                          event_type_vocabulary=event_type_vocabulary)

        event_collate = EventCollate(word_vocabulary=pretrained_word_vocabulary,
                                     event_type_vocabulary=event_type_vocabulary,
                                     entity_tag_vocabulary=entity_tag_vocabulary,
                                     sentence_max_len=512)
        train_data_loader = DataLoader(dataset=train_dataset,
                                       batch_size=batch_size,
                                       num_workers=0,
                                       collate_fn=event_collate)

        validation_data_loader = DataLoader(dataset=validation_dataset,
                                            batch_size=batch_size,
                                            num_workers=0,
                                            collate_fn=event_collate)

        if train_type == Train.NEW_TRAIN:
            trainer.train(train_data_loader=train_data_loader,
                          validation_data_loader=validation_data_loader)
        else:
            trainer.recovery_train(train_data_loader=train_data_loader,
                          validation_data_loader=validation_data_loader)


class EventConfigFactory(ConfigFactory):
    """
    事件训练的 config 工厂
    """

    def __init__(self, debug: bool = False):
        """
        初始化 config factory
        :param debug: True: 使用debug的数据，只有几条; False: 实际训练的数据
        """
        self._debug = debug

    def create(self) -> Dict:
        serialize_dir = os.path.join(ROOT_PATH,
                                     "data/event/event_detection_without_tirgger/train")
        vocabulary_dir = os.path.join(ROOT_PATH,
                                      "data/event/event_detection_without_tirgger/vocabulary")
        pretrained_embedding_file_path = os.path.join(ROOT_PATH,
                                                      "data/pretrained/glove/glove.840B.300d.txt")

        if self._debug:
            # 实际使用的是 ace 数据集, 由于 ace 数据是非公开的。所以当没有 ace 数据集，可以使用 sample 数据
            # sample 文件表示了实际训练的结构等信息，是从全量的 ace 数据中抽出的几条
            train_dataset_file_path = \
                "data/event/event_detection_without_tirgger/tests/training_data_sample.txt"
        else:
            train_dataset_file_path = "data/ace_english_event/train.txt"

        train_dataset_file_path = os.path.join(ROOT_PATH,
                                               train_dataset_file_path)

        if not os.path.isfile(train_dataset_file_path):
            train_dataset_file_path = "data/event/event_detection_without_tirgger/tests/training_data_sample.txt"
            train_dataset_file_path = os.path.join(ROOT_PATH,
                                                   train_dataset_file_path)

        if self._debug:
            validation_dataset_file_path = \
                "data/event/event_detection_without_tirgger/tests/training_data_sample.txt"
        else:
            validation_dataset_file_path = "data/ace_english_event/dev.txt"

        validation_dataset_file_path = os.path.join(ROOT_PATH, validation_dataset_file_path)

        if not os.path.isfile(validation_dataset_file_path):
            validation_dataset_file_path = "data/event/event_detection_without_tirgger/tests/training_data_sample.txt"
            validation_dataset_file_path = os.path.join(ROOT_PATH,
                                                        validation_dataset_file_path)

        pretrained_embedding_max_size = None

        if self._debug:
            # debug 模式下仅仅载入 1000 行
            pretrained_embedding_max_size = 1000

        if torch.cuda.is_available():
            cuda = [2]
        else:
            cuda = None

        config = {
            "serialize_dir": serialize_dir,
            "vocabulary_dir": vocabulary_dir,
            "train_dataset_file_path": train_dataset_file_path,
            "validation_dataset_file_path": validation_dataset_file_path,
            "pretrained_embedding_file_path": pretrained_embedding_file_path,
            "pretrained_embedding_max_size": pretrained_embedding_max_size,
            "fine_tuning": False,  # 预训练的是否进行 fine tuning
            "word_embedding_dim": 300,  # 这是与 pretrained_embedding_file_path 一致的
            "epoch": 20,
            "batch_size": 64,
            "cuda": cuda

        }
        return config


if __name__ == '__main__':

    log_util.config()

    config = EventConfigFactory(debug=True).create()

    Train()(config=config, train_type=Train.NEW_TRAIN)
