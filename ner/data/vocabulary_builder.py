#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
vocabulary 构建器

Authors: PanXu
Date:    2020/11/01 16:45:00
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from easytext.component.register import ComponentRegister
from easytext.data import Vocabulary, PretrainedVocabulary, PretrainedWordEmbeddingLoader, LabelVocabulary
from easytext.component import Component


@ComponentRegister.register(name_space="ner")
class VocabularyBuilder(Component):
    """
    Vocabulary 构建器
    """

    def __init__(self,
                 is_training: bool,
                 dataset: Dataset,
                 vocabulary_collate,
                 token_vocabulary_dir: str,
                 label_vocabulary_dir: str,
                 is_build_token_vocabulary: bool,
                 pretrained_word_embedding_loader: PretrainedWordEmbeddingLoader):

        super().__init__(is_training=is_training)

        token_vocabulary = None
        label_vocabulary = None

        if is_training:
            data_loader = DataLoader(dataset=dataset,
                                     batch_size=100,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=vocabulary_collate)
            batch_tokens = list()
            batch_sequence_labels = list()

            for collate_dict in data_loader:
                batch_tokens.extend(collate_dict["tokens"])
                batch_sequence_labels.extend(collate_dict["sequence_labels"])

            if is_build_token_vocabulary:
                token_vocabulary = Vocabulary(tokens=batch_tokens,
                                              padding=Vocabulary.PADDING,
                                              unk=Vocabulary.UNK,
                                              special_first=True)

                if pretrained_word_embedding_loader is not None:
                    token_vocabulary = \
                        PretrainedVocabulary(vocabulary=token_vocabulary,
                                             pretrained_word_embedding_loader=pretrained_word_embedding_loader)

                if token_vocabulary_dir:
                    token_vocabulary.save_to_file(token_vocabulary_dir)

            label_vocabulary = LabelVocabulary(labels=batch_sequence_labels,
                                               padding=LabelVocabulary.PADDING)

            if label_vocabulary_dir:
                label_vocabulary.save_to_file(label_vocabulary_dir)
        else:
            if is_build_token_vocabulary and token_vocabulary_dir:
                token_vocabulary = Vocabulary.from_file(token_vocabulary_dir)

            if label_vocabulary_dir:
                label_vocabulary = LabelVocabulary.from_file(label_vocabulary_dir)

        self.token_vocabulary = token_vocabulary
        self.label_vocabulary = label_vocabulary

    @staticmethod
    @ComponentRegister.register(name_space="ner")
    def label_vocabulary(vocabulary_builder: "VocabularyBuilder") -> LabelVocabulary:
        return vocabulary_builder.label_vocabulary

    @staticmethod
    @ComponentRegister.register(name_space="ner")
    def token_vocabulary(vocabulary_builder: "VocabularyBuilder") -> Vocabulary:
        return vocabulary_builder.token_vocabulary
