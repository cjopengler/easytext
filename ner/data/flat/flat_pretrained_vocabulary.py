#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
flat pretrained vocabulary

Authors: PanXu
Date:    2021/02/24 10:36:00
"""

from easytext.data import PretrainedVocabulary, Vocabulary


class FlatPretrainedVocabulary(PretrainedVocabulary):
    """
    Flat pretrained vocabulary，因为 flat ner 需要将，基于字的 embedding 以及 gaz word 的 预训练 合并在一起
    """

    def __init__(self,
                 character_pretrained_vocabulary: PretrainedVocabulary,
                 gaz_word_pretrained_vocabulary: PretrainedVocabulary):
        """
        初始化
        :param character_pretrained_vocabulary:
        :param gaz_word_pretrained_vocabulary:
        """

        assert character_pretrained_vocabulary.embedding_dim == gaz_word_pretrained_vocabulary.embedding_dim, \
            f"character_pretrained_vocabulary 与 gaz_word_pretrained_vocabulary embedding 维度必须相同"

        char_embedding_dict = self.__token_embedding_dict(character_pretrained_vocabulary)
        gaz_word_embedding_dict = self.__token_embedding_dict(gaz_word_pretrained_vocabulary)

        tokens = [char_embedding_dict.keys(), gaz_word_embedding_dict.keys()]
        char_embedding_dict.update(gaz_word_embedding_dict)

        embedding_dict = char_embedding_dict

        vocabulary = Vocabulary(tokens=tokens,
                                padding=Vocabulary.PADDING,
                                unk=Vocabulary.UNK,
                                special_first=True)

        super().__init__(vocabulary=vocabulary, pretrained_word_embedding_loader=None)

        self._embedding_dim = character_pretrained_vocabulary.embedding_dim
        self._init_embedding_matrix(vocabulary=self._vocabulary,
                                    embedding_dict=embedding_dict,
                                    embedding_dim=self._embedding_dim)

    def __token_embedding_dict(self, pretrained_vocabulary: PretrainedVocabulary):
        token_embedding_dict = dict()
        unk_index = pretrained_vocabulary.index(pretrained_vocabulary.unk)
        for index in range(pretrained_vocabulary.embedding_matrix.size(0)):

            if index in {pretrained_vocabulary.padding_index, unk_index}:
                # 对 padding_index 和 unk_index 来说，需要略过，因为是在后面合并后的 vocabulary 中，会重新填充
                continue

            token = pretrained_vocabulary.token(index)
            embedding = pretrained_vocabulary.embedding_matrix[index, :]
            token_embedding_dict[token] = embedding
        return token_embedding_dict
