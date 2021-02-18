#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
基于 bilstm + gat 的实体识别模型

相关论文:
Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network

[ACL2019 论文地址](https://www.aclweb.org/anthology/D19-1396/)


Authors: PanXu
Date:    2021/02/15 15:37:00
"""

from typing import Dict
import torch
from torch.nn import Module, Dropout, Embedding, Linear, LSTM, Parameter

from easytext.utils import bio as BIO
from easytext.data import Vocabulary, PretrainedVocabulary, LabelVocabulary
from easytext.modules import DynamicRnn, ConditionalRandomField
from easytext.component.register import ComponentRegister
from easytext.modules import GAT
from easytext.modules.seq2seq import RnnSeq2Seq

from ner.models import NerModelOutputs


class MFunsionLayer(Module):
    """
    对应原论文代码中 strategy == "m"
    将所有的 encoding 结果拼接，再进行一次线性变换
    """

    def __init__(self, label_size: int):
        super().__init__()
        self.weight = Linear(label_size * 4, label_size)

    def reset_parameters(self):
        pass

    def forward(self,
                lstm_encoding: torch.Tensor,
                c_graph_encoding: torch.Tensor,
                t_graph_encoding: torch.Tensor,
                l_graph_encoding: torch.Tensor) -> torch.Tensor:
        """
        运行
        :param lstm_encoding: lstm encoding 结果
        :param c_graph_encoding: c graph encoding 结果
        :param t_graph_encoding: t graph encoding 结果
        :param l_graph_encoding: l graph encoding 结果
        :return: 融合后的结果, shape: (B, seq_len, label_size)
        """
        encoding = torch.cat([lstm_encoding, c_graph_encoding, t_graph_encoding, l_graph_encoding], dim=-1)
        return self.weight(encoding)


class VFusionLayer(Module):
    """
    对应原论文代码中 strategy == "m"
    相当于将每一个 encoding 结果，乘以一个权重参数，然后加起来
    """

    def __init__(self, label_size: int):
        super().__init__()
        self.weight_lstm = Parameter(torch.ones(label_size))
        self.weight_c_graph = Parameter(torch.ones(label_size))
        self.weight_t_graph = Parameter(torch.ones(label_size))
        self.weight_l_graph = Parameter(torch.ones(label_size))

    def forward(self,
                lstm_encoding: torch.Tensor,
                c_graph_encoding: torch.Tensor,
                t_graph_encoding: torch.Tensor,
                l_graph_encoding: torch.Tensor) -> torch.Tensor:
        """
        运行
        :param lstm_encoding: lstm encoding 结果
        :param c_graph_encoding: c graph encoding 结果
        :param t_graph_encoding: t graph encoding 结果
        :param l_graph_encoding: l graph encoding 结果
        :return: 融合后的结果, shape: (B, seq_len, label_size)
        """
        encoding = torch.mul(lstm_encoding, self.weight_lstm) \
                   + torch.mul(c_graph_encoding, self.weight_c_graph) \
                   + torch.mul(t_graph_encoding, self.weight_t_graph) \
                   + torch.mul(l_graph_encoding, self.weight_l_graph)
        return encoding

    def reset_parameters(self):
        pass


class NFusionLayer(Module):
    """
    对应原论文代码中 strategy == "n"
    相当于将每一个 encoding 结果，乘以权重参数，然后加起来
    """

    def __init__(self):
        super().__init__()
        self.weight_lstm = Parameter(torch.ones(1))
        self.weight_c_graph = Parameter(torch.ones(1))
        self.weight_t_graph = Parameter(torch.ones(1))
        self.weight_l_graph = Parameter(torch.ones(1))

    def forward(self,
                lstm_encoding: torch.Tensor,
                c_graph_encoding: torch.Tensor,
                t_graph_encoding: torch.Tensor,
                l_graph_encoding: torch.Tensor) -> torch.Tensor:
        """
        运行
        :param lstm_encoding: lstm encoding 结果
        :param c_graph_encoding: c graph encoding 结果
        :param t_graph_encoding: t graph encoding 结果
        :param l_graph_encoding: l graph encoding 结果
        :return: 融合后的结果, shape: (B, seq_len, label_size)
        """
        encoding = torch.mul(lstm_encoding, self.weight_lstm) \
                   + torch.mul(c_graph_encoding, self.weight_c_graph) \
                   + torch.mul(t_graph_encoding, self.weight_t_graph) \
                   + torch.mul(l_graph_encoding, self.weight_l_graph)
        return encoding

    def reset_parameters(self):
        pass


@ComponentRegister.register(name_space="ner")
class BiLstmGAT(Module):
    """
    BiLstm GAT 模型

    相关论文:
    Leverage Lexical Knowledge for Chinese Named Entity Recognition via Collaborative Graph Network
    [ACL2019 论文地址](https://www.aclweb.org/anthology/D19-1396/)
    """

    def __init__(self,
                 token_vocabulary: Vocabulary,
                 token_embedding_dim: int,
                 token_embedding_dropout: float,
                 gaz_vocabulary: PretrainedVocabulary,
                 gaz_word_embedding_dropout: float,
                 num_lstm_layer: int,
                 lstm_hidden_size: int,
                 gat_hidden_size: int,
                 gat_num_heads: int,
                 gat_dropout: float,
                 lstm_dropout: float,
                 alpha: float,
                 fusion_strategy: str,
                 label_vocabulary: LabelVocabulary):

        super().__init__()

        assert gaz_vocabulary.embedding_dim == lstm_hidden_size * 2, \
            f"gaz_vocabulary.embedding_dim: {gaz_vocabulary.embedding_dim} " \
            f"与 lstm_hidden_size * 2: {lstm_hidden_size * 2} 不相等, 因为二者都会作为图的节点，所以 size 必须一致"

        self.token_vocabulary = token_vocabulary
        self.label_vocabulary = label_vocabulary

        self.token_embedding_dropout = Dropout(token_embedding_dropout)

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
        self.gaz_word_embedding_dropout = Dropout(gaz_word_embedding_dropout)

        # bilstm
        bilstm = DynamicRnn(rnn=LSTM(input_size=token_embedding_dim,
                                     hidden_size=lstm_hidden_size,
                                     num_layers=num_lstm_layer,
                                     batch_first=True,
                                     bidirectional=True))
        self.bilstm_seq2seq = RnnSeq2Seq(bilstm)
        self.lstm_dropout = Dropout(lstm_dropout)
        self.lstm_encoding_feed_forward = Linear(in_features=lstm_hidden_size * 2,
                                                 out_features=self.label_vocabulary.label_size)
        # C-Graph
        self.c_gat = GAT(in_features=2 * lstm_hidden_size,
                         out_features=label_vocabulary.label_size,
                         dropout=gat_dropout,
                         alpha=alpha,
                         num_heads=gat_num_heads,
                         hidden_size=gat_hidden_size)

        # T-Graph
        self.t_gat = GAT(in_features=2 * lstm_hidden_size,
                         out_features=label_vocabulary.label_size,
                         dropout=gat_dropout,
                         alpha=alpha,
                         num_heads=gat_num_heads,
                         hidden_size=gat_hidden_size)

        # L-Graph
        self.l_gat = GAT(in_features=2 * lstm_hidden_size,
                         out_features=label_vocabulary.label_size,
                         dropout=gat_dropout,
                         alpha=alpha,
                         num_heads=gat_num_heads,
                         hidden_size=gat_hidden_size)

        if fusion_strategy == "m":
            self.fusion_layer = MFunsionLayer(label_size=label_vocabulary.label_size)
        elif fusion_strategy == "v":
            self.fusion_layer = VFusionLayer(label_size=label_vocabulary.label_size)
        elif fusion_strategy == "n":
            self.fusion_layer = NFusionLayer()
        else:
            raise RuntimeError(f"fusion_stategy 必须是: m, v, n 之一, 而现在是 {fusion_strategy}")
        # crf
        constraints = BIO.allowed_transitions(label_vocabulary=self.label_vocabulary)
        self.crf = ConditionalRandomField(num_tags=self.label_vocabulary.label_size,
                                          constraints=constraints)

    def reset_parameters(self):
        pass

    def forward(self,
                tokens: torch.Tensor,
                gaz_words: torch.Tensor,
                t_graph: torch.Tensor,
                c_graph: torch.Tensor,
                l_graph: torch.Tensor,
                metadata: Dict) -> NerModelOutputs:

        # 计算 token lstm
        mask = (tokens != self.token_vocabulary.padding_index)
        bool_mask = mask.bool()
        token_embeddings = self.token_embedding(tokens)
        token_embeddings = self.token_embedding_dropout(token_embeddings)

        bilstm_seq_encoding = self.bilstm_seq2seq(sequence=token_embeddings, mask=bool_mask)
        bilstm_seq_encoding = self.lstm_dropout(bilstm_seq_encoding)
        bilstm_seq_encoding = self.lstm_encoding_feed_forward(bilstm_seq_encoding)

        seq_len = bilstm_seq_encoding.size(1)

        # gaz word embedding
        gaz_word_embeddings = self.gaz_word_embedding(gaz_words)
        gaz_word_embeddings = self.gaz_word_embedding_dropout(gaz_word_embeddings)

        # 将 bilstm_seq_encoding 与 gaz_word_embeddings 组成 nodes
        nodes = torch.cat([bilstm_seq_encoding, gaz_word_embeddings], dim=1)

        # 计算图注意力
        c_graph_encoding = self.c_gat(nodes=nodes, adj=c_graph)
        c_graph_encoding = c_graph_encoding[:, :seq_len, :]
        t_graph_encoding = self.t_gat(nodes=nodes, adj=t_graph)
        t_graph_encoding = t_graph_encoding[:, :seq_len, :]
        l_graph_encoding = self.l_gat(nodes=nodes, adj=l_graph)
        l_graph_encoding = l_graph_encoding[:, :seq_len, :]

        # fusion, 融合
        logits = self.fusion_layer(lstm_encoding=bilstm_seq_encoding,
                                   c_graph_encoding=c_graph_encoding,
                                   t_graph_encoding=t_graph_encoding,
                                   l_graph_encoding=l_graph_encoding)

        return NerModelOutputs(logits=logits, mask=mask, crf=self.crf)
