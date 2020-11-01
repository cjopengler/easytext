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


@ComponentRegister.register_class(name="VocabularyBuilder", name_space="data")
class VocabularyBuilder:
    """
    Vocabulary 构建器
    """

    def __init__(self,
                 dataset: Dataset,
                 vocabulary_collate,
                 is_build_token_vocabulary: bool,
                 pretrained_word_embedding_loader: PretrainedWordEmbeddingLoader):

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

        label_vocabulary = LabelVocabulary(labels=batch_sequence_labels,
                                           padding=LabelVocabulary.PADDING)

        self.token_vocabulary = token_vocabulary
        self.label_vocabulary = label_vocabulary

