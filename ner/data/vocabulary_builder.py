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
        """
        词汇表构建器
        :param is_training: 因为在 train 和 非 train, 词汇表的构建行为有所不同;
                            如果是 train, 则一般需要重新构建; 而对于 非train, 使用先前构建好的即可。
        :param dataset: 数据集
        :param vocabulary_collate: 词汇表 collate
        :param token_vocabulary_dir: token vocabulary 存放目录
        :param label_vocabulary_dir: label vocabulary 存放目录
        :param is_build_token_vocabulary: 是否构建 token vocabulary, 因为在使用 Bert 或者 其他模型作为预训练的 embedding,
                                          则没有必要构建 token vocabulary.
        :param pretrained_word_embedding_loader: 预训练词汇表
        """

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
