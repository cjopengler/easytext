#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
模型训练

Authors: PanXu
Date:    2020/07/18 18:32:00
"""
from typing import Dict

import torch

from torch.utils.data import DataLoader

from easytext.trainer import Trainer
from easytext.data import Vocabulary, PretrainedVocabulary, LabelVocabulary
from easytext.utils import log_util
from easytext.data import GloveLoader

from acsa.data.dataset import ACSASemEvalDataset
from acsa.data import VocabularyCollate
from acsa.data import ACSAModelCollate
from acsa.models import ATAELstm
from acsa.loss import ACSALoss
from acsa.metrics import ACSAModelMetric
from acsa.label_decoder import ACSALabelDecoder
from acsa.optimize import ACSAOptimizerFactory

from acsa.train.acsa_config_factory import ACSAConfigFactory


class Train:
    """
    ACSA 模型训练
    """

    def __init__(self, config: Dict):
        self.config = config

    def build_vocabulary(self):
        training_dataset_file_path = self.config["training_dataset_file_path"]

        dataset = ACSASemEvalDataset(dataset_file_path=training_dataset_file_path)

        collate_fn = VocabularyCollate()
        data_loader = DataLoader(dataset=dataset,
                                 batch_size=10,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=collate_fn)

        tokens = list()
        categories = list()
        labels = list()
        for collate_dict in data_loader:
            tokens.append(collate_dict["tokens"])
            categories.append(collate_dict["categories"])
            labels.append(collate_dict["labels"])

        token_vocabulary = Vocabulary(
            tokens=tokens,
            padding=Vocabulary.PADDING,
            unk=Vocabulary.UNK,
            special_first=True
        )

        if not self.config["debug"]:
            pretrained_file_path = self.config["pretrained_file_path"]
            pretrained_loader = GloveLoader(embedding_dim=300,
                                            pretrained_file_path=pretrained_file_path)
            pretrained_token_vocabulary = PretrainedVocabulary(
                vocabulary=token_vocabulary,
                pretrained_word_embedding_loader=pretrained_loader
            )

            token_vocabulary = pretrained_token_vocabulary

        category_vocabulary = LabelVocabulary(labels=categories, padding=None)
        label_vocabulary = LabelVocabulary(labels=labels, padding=None)

        return {"token_vocabulary": token_vocabulary,
                "category_vocabulary": category_vocabulary,
                "label_vocabulary": label_vocabulary}

    def build_model(self,
                    token_vocabulary: Vocabulary,
                    category_vocabulary: LabelVocabulary,
                    label_vocabulary: LabelVocabulary):
        model_config = self.config["model"]
        model = ATAELstm(
            token_vocabulary=token_vocabulary,
            token_embedding_dim=model_config["token_embedding_dim"],
            category_vocabulary=category_vocabulary,
            category_embedding_dim=model_config["category_embedding_dim"],
            label_vocabulary=label_vocabulary
        )

        return model

    def __call__(self):

        serialize_dir = self.config["serialize_dir"]
        num_epoches = self.config["num_epoches"]

        vocabulary_dict = self.build_vocabulary()
        token_vocabulary = vocabulary_dict["token_vocabulary"]
        category_vocabulary = vocabulary_dict["category_vocabulary"]
        label_vocabulary = vocabulary_dict["label_vocabulary"]

        model = self.build_model(token_vocabulary=token_vocabulary,
                                 category_vocabulary=category_vocabulary,
                                 label_vocabulary=label_vocabulary)

        loss = ACSALoss()
        label_decoder = ACSALabelDecoder(label_vocabulary=label_vocabulary)
        metrics = ACSAModelMetric(label_decoder=label_decoder)
        optimizer_factory = ACSAOptimizerFactory(config=self.config)

        patient = self.config["patient"]
        num_check_point_keep = self.config["num_check_point_keep"]
        trainer = Trainer(serialize_dir=serialize_dir,
                          num_epoch=num_epoches,
                          model=model,
                          loss=loss,
                          metrics=metrics,
                          patient=patient,
                          num_check_point_keep=num_check_point_keep,
                          optimizer_factory=optimizer_factory)

        training_dataset_file_path = self.config["training_dataset_file_path"]
        validation_dataset_file_path = self.config["validation_dataset_file_path"]

        train_dataset = ACSASemEvalDataset(training_dataset_file_path)
        batch_size = self.config["batch_size"]

        model_collate = ACSAModelCollate(token_vocabulary=token_vocabulary,
                                         category_vocabulary=category_vocabulary,
                                         label_vocabulary=label_vocabulary)

        train_data_loader = DataLoader(dataset=train_dataset,
                                       shuffle=True,
                                       batch_size=batch_size,
                                       num_workers=0,
                                       collate_fn=model_collate)

        validation_dataset = ACSASemEvalDataset(validation_dataset_file_path)
        validation_data_loader = DataLoader(dataset=validation_dataset,
                                            shuffle=False,
                                            batch_size=batch_size,
                                            num_workers=0,
                                            collate_fn=model_collate)

        trainer.train(train_data_loader=train_data_loader,
                      validation_data_loader=validation_data_loader)


if __name__ == '__main__':

    log_util.config()

    config = ACSAConfigFactory(debug=True).create()
    Train(config=config)()
