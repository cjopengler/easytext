#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
lattice mode collate

Authors: PanXu
Date:    2021/02/08 11:40:00
"""

from typing import List

import torch

from easytext.data import ModelCollate
from easytext.data import Vocabulary
from easytext.data import LabelVocabulary
from easytext.data import Instance
from easytext.data import ModelInputs

from ner.data.lattice import Gazetteer


class LatticeModelCollate(ModelCollate):
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








