#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
lattice vocabulary builder

Authors: PanXu
Date:    2021/02/09 09:23:00
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from easytext.component.register import ComponentRegister
from easytext.data import Vocabulary, PretrainedVocabulary, PretrainedWordEmbeddingLoader, LabelVocabulary
from easytext.component import Component

from ner.data import VocabularyBuilder
from ner.data.lattice import Gazetteer
from ner.data.lattice import GazVocabularyCollate


@ComponentRegister.register(name_space="lattice")
class GazVocabularyBuilder(Component):
    """
    Gaz Vocabulary Builder
    """

    def __init__(self,
                 is_training: bool,
                 dataset: Dataset,
                 gaz_vocabulary_dir: str,
                 gaz_pretrained_word_embedding_loader: PretrainedWordEmbeddingLoader):
        """
        构建 Gaz 词汇表
        :param is_training: 当前是否 Training 状态
        :param dataset: 数据集
        :param gaz_vocabulary_dir: gaz 词汇表存放目录
        :param gaz_pretrained_word_embedding_loader: gaz 预训练 word embedding 载入器
        """

        super().__init__(is_training=is_training)

        # gazetter 理论上来说，应该支持持久化的，这里并没有做
        gazetteer = Gazetteer(gaz_pretrained_word_embedding_loader=gaz_pretrained_word_embedding_loader)

        if is_training:
            gaz_vocabulary_collate = GazVocabularyCollate(gazetteer=gazetteer)

            data_loader = DataLoader(dataset=dataset,
                                     batch_size=100,
                                     shuffle=False,
                                     num_workers=0,
                                     collate_fn=gaz_vocabulary_collate)

            gaz_words = list()

            for batch_gaz_words in data_loader:
                gaz_words.extend(batch_gaz_words)

                gaz_vocabulary = Vocabulary(tokens=gaz_words,
                                            padding=Vocabulary.PADDING,
                                            unk=Vocabulary.UNK,
                                            special_first=True)

                gaz_vocabulary = PretrainedVocabulary(
                    vocabulary=gaz_vocabulary,
                    pretrained_word_embedding_loader=gaz_pretrained_word_embedding_loader)

                gaz_vocabulary.save_to_file(gaz_vocabulary_dir)
        else:

            gaz_vocabulary = Vocabulary.from_file(gaz_vocabulary_dir)

        self.gaz_vocabulary = gaz_vocabulary
        self.gazetteer = gazetteer

    @staticmethod
    @ComponentRegister.register(name_space="lattice")
    def gaz_vocabulary(gaz_vocabulary_builder: "GazVocabularyBuilder") -> Vocabulary:
        return gaz_vocabulary_builder.gaz_vocabulary

    @staticmethod
    @ComponentRegister.register(name_space="lattice")
    def gazetteer(gaz_vocabulary_builder: "GazVocabularyBuilder") -> Gazetteer:
        return gaz_vocabulary_builder.gazetteer
