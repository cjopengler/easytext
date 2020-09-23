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

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from transformers import BertTokenizer

from easytext.trainer import Trainer
from easytext.data import Vocabulary, LabelVocabulary, PretrainedVocabulary
from easytext.data import GloveLoader
from easytext.utils import log_util

from ner.models import NerV1, NerV2, NerV3, NerV4
from ner.data.dataset import Conll2003Dataset, MsraDataset
from ner.data import VocabularyCollate
from ner.data import NerModelCollate, BertModelCollate
from ner.loss import NerLoss
from ner.loss import NerCRFLoss
from ner.metrics import NerModelMetricAdapter
from ner.optimizer import NerOptimizerFactory, NerV4BertOptimizerFactory
from ner.label_decoder import NerMaxModelLabelDecoder
from ner.label_decoder import NerCRFModelLabelDecoder
from ner.config import NerConfigFactory


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

        model_name = self.config["model_name"]

        token_vocabulary = None

        if model_name in {NerV1.NAME, NerV2.NAME, NerV3.NAME}:
            token_vocabulary = Vocabulary(tokens=batch_tokens,
                                          padding=Vocabulary.PADDING,
                                          unk=Vocabulary.UNK,
                                          special_first=True)

        if model_name in {NerV2.NAME, NerV3.NAME}:
            pretrained_word_embedding_file_path = self.config["pretrained_word_embedding_file_path"]
            glove_loader = GloveLoader(embedding_dim=100,
                                       pretrained_file_path=pretrained_word_embedding_file_path)

            token_vocabulary = PretrainedVocabulary(vocabulary=token_vocabulary,
                                                    pretrained_word_embedding_loader=glove_loader)

        if model_name in {NerV4}:
            token_vocabulary = None

        label_vocabulary = LabelVocabulary(labels=batch_sequence_labels,
                                           padding=LabelVocabulary.PADDING)

        return {"token_vocabulary": token_vocabulary,
                "label_vocabulary": label_vocabulary}

    def build_model(self,
                    token_vocabulary: Union[Vocabulary, PretrainedVocabulary],
                    label_vocabulary: LabelVocabulary):
        model_name = config["model_name"]

        model_config = config["model"]

        if model_name == NerV1.NAME:
            model = NerV1(token_vocabulary=token_vocabulary,
                          label_vocabulary=label_vocabulary,
                          word_embedding_dim=model_config["word_embedding_dim"],
                          hidden_size=model_config["hidden_size"],
                          num_layer=model_config["num_layer"],
                          dropout=model_config["dropout"])
        elif model_name == NerV2.NAME:
            model = NerV2(token_vocabulary=token_vocabulary,
                          label_vocabulary=label_vocabulary,
                          word_embedding_dim=model_config["word_embedding_dim"],
                          hidden_size=model_config["hidden_size"],
                          num_layer=model_config["num_layer"],
                          dropout=model_config["dropout"])
        elif model_name == NerV3.NAME:
            model = NerV3(token_vocabulary=token_vocabulary,
                          label_vocabulary=label_vocabulary,
                          word_embedding_dim=model_config["word_embedding_dim"],
                          hidden_size=model_config["hidden_size"],
                          num_layer=model_config["num_layer"],
                          dropout=model_config["dropout"])
        elif model_name == NerV4.NAME:
            model = NerV4(bert_dir=model_config["bert_dir"],
                          label_vocabulary=label_vocabulary,
                          dropout=model_config["dropout"],
                          is_used_crf=model_config["is_used_crf"])
        else:
            raise RuntimeError(f"model name: {model_name} 是非法的!")

        return model

    def build_loss(self, label_vocabulary: LabelVocabulary):
        model_name = self.config["model_name"]
        model_config = self.config["model"]

        if model_name == NerV1.NAME:
            loss = NerLoss()
        elif model_name == NerV2.NAME:
            loss = NerLoss()
        elif model_name == NerV3.NAME:
            loss = NerCRFLoss(is_used_crf=model_config["is_used_crf"],
                              label_vocabulary=label_vocabulary)
        elif model_name == NerV4.NAME:
            loss = NerCRFLoss(is_used_crf=model_config["is_used_crf"],
                              label_vocabulary=label_vocabulary)

        return loss

    def build_model_metric(self, label_vocabulary: LabelVocabulary):
        model_name = self.config["model_name"]

        if model_name == NerV1.NAME:
            metric = NerModelMetricAdapter(label_vocabulary=label_vocabulary,
                                           model_label_decoder=NerMaxModelLabelDecoder(
                                               label_vocabulary=label_vocabulary))
        elif model_name == NerV2.NAME:
            metric = NerModelMetricAdapter(label_vocabulary=label_vocabulary,
                                           model_label_decoder=NerMaxModelLabelDecoder(
                                               label_vocabulary=label_vocabulary))
        elif model_name == NerV3.NAME:
            metric = NerModelMetricAdapter(label_vocabulary=label_vocabulary,
                                           model_label_decoder=NerCRFModelLabelDecoder(
                                               label_vocabulary=label_vocabulary))
        elif model_name == NerV4.NAME:
            model_config = self.config["model"]
            is_used_crf = model_config["is_used_crf"]

            if is_used_crf:
                metric = NerModelMetricAdapter(label_vocabulary=label_vocabulary,
                                               model_label_decoder=NerCRFModelLabelDecoder(
                                                   label_vocabulary=label_vocabulary))
            else:
                metric = NerModelMetricAdapter(label_vocabulary=label_vocabulary,
                                               model_label_decoder=NerMaxModelLabelDecoder(
                                                   label_vocabulary=label_vocabulary))
        else:
            raise RuntimeError(f"model name: {model_name} 是非法的!")

        return metric

    def build_model_collate(self, token_vocabulary: Vocabulary, label_vocabulary: LabelVocabulary):
        model_name = self.config["model_name"]

        if model_name in {NerV1.NAME, NerV2.NAME, NerV3.NAME}:
            model_collate = NerModelCollate(token_vocab=token_vocabulary,
                                            sequence_label_vocab=label_vocabulary,
                                            sequence_max_len=512)
        elif model_name == NerV4.NAME:
            bert_dir = self.config["bert_dir"]
            bert_tokenizer = BertTokenizer.from_pretrained(bert_dir)
            model_collate = BertModelCollate(tokenizer=bert_tokenizer,
                                             sequence_label_vocab=label_vocabulary,
                                             sequence_max_len=508)
        else:
            raise RuntimeError(f"model name: {model_name} 非法")
        return model_collate

    def build_optimizer(self):

        model_name = self.config["model_name"]
        fine_tuning = self.config["fine_tuning"]

        if model_name in {NerV1.NAME, NerV2.NAME, NerV3.NAME}:
            optimizer_factory = NerOptimizerFactory(fine_tuning=fine_tuning)
        elif model_name == NerV4.NAME:
            optimizer_factory = NerV4BertOptimizerFactory(fine_tuning=fine_tuning)

        return optimizer_factory

    def __call__(self):
        config = self.config
        serialize_dir = config["serialize_dir"]

        dataset_name = config["dataset_name"]
        model_name = config["model_name"]

        if self._train_type == Train.NEW_TRAIN:
            # 清理 serialize dir
            if os.path.isdir(serialize_dir):
                shutil.rmtree(serialize_dir)
                os.makedirs(serialize_dir)

        num_epoch = config["num_epoch"]
        patient = config["patient"]
        num_check_point_keep = config["num_check_point_keep"]

        if dataset_name == Conll2003Dataset.NAME:
            train_dataset = Conll2003Dataset(dataset_file_path=config["train_dataset_file_path"])
            validation_dataset = Conll2003Dataset(dataset_file_path=config["validation_dataset_file_path"])
        elif dataset_name == MsraDataset.NAME:
            train_dataset = MsraDataset(dataset_file_path=config["train_dataset_file_path"])
            validation_dataset = MsraDataset(dataset_file_path=config["validation_dataset_file_path"])
        else:
            raise RuntimeError(f"dataset name: {dataset_name} 非法")

        # 构建 vocabulary
        vocab_dict = self.build_vocabulary(dataset=train_dataset)
        token_vocabulary = vocab_dict["token_vocabulary"]
        label_vocabulary = vocab_dict["label_vocabulary"]

        model = self.build_model(token_vocabulary=token_vocabulary,
                                 label_vocabulary=label_vocabulary)

        loss = self.build_loss(label_vocabulary=label_vocabulary)

        metric = self.build_model_metric(label_vocabulary=label_vocabulary)
        optimizer_factory = self.build_optimizer()

        cuda = config["cuda"]

        trainer = Trainer(serialize_dir=serialize_dir,
                          num_epoch=num_epoch,
                          model=model,
                          loss=loss,
                          metrics=metric,
                          optimizer_factory=optimizer_factory,
                          lr_scheduler_factory=None,
                          patient=patient,
                          num_check_point_keep=num_check_point_keep,
                          cuda_devices=cuda)

        model_collate = self.build_model_collate(token_vocabulary=token_vocabulary,
                                                 label_vocabulary=label_vocabulary)
        train_data_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.config["train_batch_size"],
            shuffle=True,
            num_workers=0,
            collate_fn=model_collate
        )

        validation_data_loader = DataLoader(
            dataset=validation_dataset,
            batch_size=self.config["test_batch_size"],
            shuffle=False,
            num_workers=0,
            collate_fn=model_collate
        )

        trainer.train(train_data_loader=train_data_loader,
                      validation_data_loader=validation_data_loader)


if __name__ == '__main__':
    log_util.config(level=logging.INFO)

    config = NerConfigFactory(debug=True,
                              model_name=NerV4.NAME,
                              dataset_name=MsraDataset.NAME).create()
    Train(train_type=Train.NEW_TRAIN, config=config)()
