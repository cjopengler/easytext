#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
brief

Authors: PanXu
Date:    2021/02/15 18:18:00
"""

from typing import List, Tuple
import numpy as np

import torch

from easytext.data import ModelCollate
from easytext.data import Vocabulary
from easytext.data import LabelVocabulary
from easytext.data import Instance
from easytext.data import ModelInputs
from easytext.component.register import ComponentRegister

from ner.data.lattice import Gazetteer


def graph_generator(max_sequence_len: int,
                    max_gaz_words_len: int,
                    gaz_ids: List[List[List[int]]]) -> Tuple:
    """
    创建 C, T, L graph
    :param max_sequence_len: 在 batch 中的 sequence 的最大长度， 因为 会padding
    :param max_gaz_words_len: 在 batch 中的 gaz words 的最大长度, 因为会 padding
    :param gaz_ids: 句子匹配上的 gaz id, 这是 3 维 List, 长度与句子的实际长度是一样的，是句子的实际长度不是 max_sequence_len。
                    在第 0 维，与句子的 "字" 的 index 是一一对应的; 第 1 维, 表示的是 放置的是 对应的 gaz word id 列表，以及
                    当前 gaz word id 匹配从当前句子中字 index 的词的长度。例如:
                    [[], [[25,13],[2,4]], [], [[33], [2]], []], 表示在字序列中，第 2个 字，所对应的词 id 是 25 和13 , 对应的长度是 2 和 4。
                    例如: "到 长 江 大 桥", 该序列长度是 5， 所以 skip_input 也是 5, 其中 "长" index=1,
                    对应 "长江" 和 "长江大桥", 其中 "长江" 在词汇表中的 id 是25, 长度是 2;
                    "长江大桥" 对应词汇表中 id 是 13， 长度是 4;
                    同样 "大桥", 对应 词汇表 id 33, 长度是 2.
    :return: C, T, L graph 对应的邻接矩阵
    """
    gaz_seq = []

    # gaz_ids 就是实际句子的长度
    sentence_len = len(gaz_ids)

    # gaz word 中，非空的长度，也就是实际的 gaz words 的数量
    gaz_len = 0
    for ele in gaz_ids:
        if ele:
            gaz_len += len(ele[0])

    # matrix 是邻接矩阵，因为需要将 sequence 和 gaz words 拼接，所以使用二者相加
    matrix_size = max_gaz_words_len + max_sequence_len

    # 这些邻接矩阵都是包含 self-loop 的图
    t_matrix = np.eye(matrix_size, dtype=int)
    l_matrix = np.eye(matrix_size, dtype=int)
    c_matrix = np.eye(matrix_size, dtype=int)

    # 因为 T, L 包含了句子本身字与下一个字的相连，所以用下面这种方式先填充
    # 也就是: M[0,1]=M[1,0]=M[1,2]=M[2,1]=...=1
    add_matrix1 = np.zeros((matrix_size, matrix_size), dtype=int)
    add_matrix2 = np.zeros((matrix_size, matrix_size), dtype=int)
    add_matrix1[:sentence_len, :sentence_len] = np.eye(sentence_len, k=1, dtype=int)
    add_matrix2[:sentence_len, :sentence_len] = np.eye(sentence_len, k=-1, dtype=int)
    t_matrix = t_matrix + add_matrix1 + add_matrix2
    l_matrix = l_matrix + add_matrix1 + add_matrix2

    # 下面将 gaz word 计算进去，gaz word 的 index 是在 max_sequence_len 之后的
    index = max_sequence_len

    for i in range(sentence_len):

        # 将当前句子中 i 的 gaz word 获取到, 用 word_id 来作为拼接后 gaz word 的 新的id
        # 主要用来计算 T-Graph 中的 word 和 word 相连
        if gaz_ids[i]:
            word_id[i] = [0] * len(gaz_ids[i][1])

            for j in range(len(gaz_ids[i][1])):
                word_id[i][j] = index
                index = index + 1

    index_gaz = max_sequence_len
    index_char = 0

    for k in range(len(gaz_ids)):
        ele = gaz_ids[k]

        if ele:

            for i in range(len(ele[0])):
                # 每一个 gaz word id
                gaz_seq.append(ele[0][i])

                # 设置 L-graph, 当前 gaz word 有两个需要设置
                l_matrix[index_gaz, index_char] = 1
                l_matrix[index_char, index_gaz] = 1
                l_matrix[index_gaz, index_char + ele[1][i] - 1] = 1
                l_matrix[index_char + ele[1][i] - 1, index_gaz] = 1

                # 设置 C-graph
                for m in range(ele[1][i]):
                    c_matrix[index_gaz, index_char + m] = 1
                    c_matrix[index_char + m, index_gaz] = 1

                # 设置 T-graph
                if index_char > 0:
                    t_matrix[index_gaz, index_char - 1] = 1
                    t_matrix[index_char - 1, index_gaz] = 1

                    if index_char + ele[1][i] < sentence_len:
                        t_matrix[index_gaz, index_char + ele[1][i]] = 1
                        t_matrix[index_char + ele[1][i], index_gaz] = 1
                else:
                    t_matrix[index_gaz, index_char + ele[1][i]] = 1
                    t_matrix[index_char + ele[1][i], index_gaz] = 1

                # T-Graph 中的 word 和 word 相连
                if index_char + ele[1][i] < sentence_len:
                    if gaz_ids[index_char + ele[1][i]]:
                        for p in range(len(gaz_ids[index_char + ele[1][i]][1])):
                            q = word_id[index_char + ele[1][i]][p]
                            t_matrix[index_gaz, q] = 1
                            t_matrix[q, index_gaz] = 1
                index_gaz = index_gaz + 1
        index_char = index_char + 1

    return t_matrix, c_matrix, l_matrix


@ComponentRegister.register(name_space="bilstm_gat")
class BiLstmGATModelCollate(ModelCollate):
    """
    Lattice Model Collate
    """

    def __init__(self,
                 token_vocabulary: Vocabulary,
                 gazetter: Gazetteer,
                 gaz_vocabulary: Vocabulary,
                 label_vocabulary: LabelVocabulary):
        self._token_vocabulary = token_vocabulary
        self._gazetteer = gazetter
        self._gaz_vocabulary = gaz_vocabulary
        self._label_vocabulary = label_vocabulary

    def __call__(self, instances: List[Instance]) -> ModelInputs:
        batch_max_len = 0
        batch_size = 0

        batch_token_indices = list()
        batch_gaz_word_indices = list()
        batch_sequence_label_indices = None
        batch_metadatas = list()

        for instance in iter(instances):
            tokens = instance["tokens"]

            if len(tokens) > batch_max_len:
                batch_max_len = len(tokens)

        for instance in instances:
            metadata = dict()
            batch_metadatas.append(metadata)

            batch_size += 1
            # 对 token 进行 index
            token_indices = [self._token_vocabulary.padding_index] * batch_max_len

            for i, token in enumerate(instance["tokens"][0: batch_max_len]):
                token_indices[i] = self._token_vocabulary.index(token.text)

            batch_token_indices.append(token_indices)

            # 产生 gaz words 已经相应的 index
            sentence = "".join([t.text for t in instance["tokens"][:batch_max_len]])

            metadata["tokens"] = sentence

            gaz_words_index_list = list()
            gaz_words_list = self._gazetteer.enumerate_match_list(sentence)

            metadata["gaz_words"] = gaz_words_list

            for gaz_words in gaz_words_list:

                if len(gaz_words) > 0:
                    # 这种情况将 word index 以及 长度添加进去
                    # :
                    word_indices = [self._gaz_vocabulary.index(gaz_word) for gaz_word in gaz_words]
                    word_lengths = [len(gaz_word) for gaz_word in gaz_words]
                    gaz_words_index_list.append([word_indices, word_lengths])
                else:
                    # 这种情况填充空的
                    gaz_words_index_list.append([])

            batch_gaz_word_indices.append(gaz_words_index_list)

            if "sequence_label" in instance:
                sequence_label = list()
                sequence_label_indices = [self._label_vocabulary.padding_index] * batch_max_len

                for i, sl in enumerate(instance["sequence_label"][0: batch_max_len]):
                    sequence_label_indices[i] = self._label_vocabulary.index(sl)
                    sequence_label.append(sl)
                metadata["sequence_label"] = " ".join(sequence_label)

                if batch_sequence_label_indices is None:
                    batch_sequence_label_indices = list()

                batch_sequence_label_indices.append(sequence_label_indices)

        batch_token_indices = torch.tensor(batch_token_indices, dtype=torch.long)

        if batch_sequence_label_indices is not None:
            batch_sequence_label_indices = torch.tensor(batch_sequence_label_indices, dtype=torch.long)

        return ModelInputs(batch_size=batch_size,
                           model_inputs={"tokens": batch_token_indices,
                                         "gaz_list": batch_gaz_word_indices,
                                         "metadata": batch_metadatas},
                           labels=batch_sequence_label_indices)









