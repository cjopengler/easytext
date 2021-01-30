#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
基于 Lattice LSTM 的 ner 模型

Authors: PanXu
Date:    2021/01/30 10:39:00
"""

import torch
from torch.nn import Module, Dropout, Embedding, Linear

from easytext.utils import bio as BIO
from easytext.data import Vocabulary, PretrainedVocabulary, LabelVocabulary
from easytext.modules import LatticeLSTM, ConditionalRandomField


class LatticeNer(Module):
    """
    基于 LatticeLstm 的 ner 识别模型
    默认使用双向的 Lattice LSTM 模型
    """

    def __init__(self,
                 token_vocabulary: Vocabulary,
                 token_embedding_dim: int,
                 token_embedding_dropout: float,
                 gaz_vocabulary: PretrainedVocabulary,
                 gaz_word_embedding_dim: int,
                 gaz_word_embedding_dropout: float,
                 label_vocabulary: LabelVocabulary,
                 hidden_size: int,
                 num_layer: int,
                 lstm_dropout: float):

        super().__init__()

        self.token_vocabulary = token_vocabulary
        self.label_vocabulary = label_vocabulary

        self.num_layer = num_layer

        self.token_embedding_dropout = Dropout(token_embedding_dropout)
        self.lstm_dropout = Dropout(lstm_dropout)

        if isinstance(self.token_vocabulary, Vocabulary):
            self.token_embedding: Embedding = Embedding(num_embeddings=self.token_vocabulary.size,
                                                        embedding_dim=token_embedding_dim,
                                                        padding_idx=self.token_vocabulary.padding_index)

        elif isinstance(self.token_vocabulary, PretrainedVocabulary):
            self.token_embedding: Embedding = Embedding.from_pretrained(self.token_vocabulary.embedding_matrix,
                                                                        freeze=True,
                                                                        padding_idx=self.token_vocabulary.padding_index)

        self.gaz_word_embedding = Embedding.from_pretrained(gaz_vocabulary.embedding_matrix,
                                                            freeze=True,
                                                            padding_idx=gaz_vocabulary.padding_index)
        # 默认使用双向的 Lattice LSTM
        # 前向 lattice lstm
        self.forward_lattice_lstm = LatticeLSTM(input_dim=token_embedding_dim,
                                                hidden_dim=hidden_size,
                                                gaz_word_embedding_dim=gaz_word_embedding_dim,
                                                gaz_word_embedding=self.gaz_word_embedding,
                                                gaz_word_embedding_dropout=gaz_word_embedding_dropout,
                                                left2right=True)

        # 反向 lattice lstm
        self.backward_lattice_lstm = LatticeLSTM(input_dim=token_embedding_dim,
                                                 hidden_dim=hidden_size,
                                                 gaz_word_embedding_dim=gaz_word_embedding_dim,
                                                 gaz_word_embedding=self.gaz_word_embedding,
                                                 gaz_word_embedding_dropout=gaz_word_embedding_dropout,
                                                 left2right=False)
        # 将 双向 lattice lstm 的输出转化到 label 空间
        self.linear = Linear(in_features=(hidden_size * 2),
                             out_features=label_vocabulary.label_size)

        # crf
        constraints = BIO.allowed_transitions(label_vocabulary=self.label_vocabulary)
        self.crf = ConditionalRandomField(num_tags=self.label_vocabulary.label_size,
                                          constraints=constraints)

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self, tokens, gaz_list, ):

        # 计算词向量
        token_embeddings = self.token_embedding(tokens)

        token_embeddings = self.token_embedding_dropout(token_embeddings)

        # 前向计算
        forward_lattice_hidden, _ = self.forward_lattice_lstm(token_embeddings, gaz_list, None)

        # 反向计算
        backward_lattice_hidden, _ = self.backward_lattice_lstm(token_embeddings, gaz_list, None)

        # 将前向计算和反向计算拼接
        lattice_hidden = torch.cat([forward_lattice_hidden, backward_lattice_hidden], 2)

        lattice_hidden = self.droplstm(lattice_hidden)
        return lattice_hidden
