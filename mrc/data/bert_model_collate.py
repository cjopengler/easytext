#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
基于 bert 模型的 向量化

Authors: PanXu
Date:    2021/10/23 11:45:00
"""
from typing import List
import torch

from transformers import BertTokenizer

from easytext.data import Instance

from mrc.data import MRCModelInputs

from easytext.component.register import ComponentRegister


@ComponentRegister.register(name_space="mrc_ner")
class BertModelCollate:
    """
    Bert Model Collate
    """

    def __init__(self, tokenizer: BertTokenizer, max_length: int = 128):
        """
        初始化
        :param max_length: 拼接后文本的最大长度，论文中是 128 这里保持不变
        """
        self._max_length = max_length
        self._tokenizer = tokenizer

    def __call__(self, instances: List[Instance]) -> MRCModelInputs:

        batch_size = len(instances)
        # 获取当前 batch 最大长度
        # 3 表示: CLS, SEP, SEP 3个 special token
        batch_max_length = max(len(instance["context"] + instance["query"]) + 3 for instance in instances)

        batch_max_length = min(batch_max_length, self._max_length)

        batch_text_pairs = [[instance["query"], instance["context"]] for instance in instances]

        batch_inputs = self._tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_text_pairs,
                                                         truncation=True,
                                                         padding=True,
                                                         max_length=batch_max_length,
                                                         return_length=True,
                                                         add_special_tokens=True,
                                                         return_special_tokens_mask=True,
                                                         # return_token_type_ids=True,
                                                         return_tensors="pt")

        batch_special_tokens_mask = batch_inputs["special_tokens_mask"]

        # 将special_tokens_mask 0->1, 1->0, 就变成了 sequence 去掉 CLS 和 SEP 的 mask 了
        batch_sequence_mask: torch.Tensor = batch_special_tokens_mask == 0

        # 需要对 非 context 中的，包括 query 以及 padding, CLS, SEP 全部 mask 掉，以便进行 label 的预测
        # 因为 label 是在 context 中的
        batch_sequence_mask = batch_sequence_mask * batch_inputs["token_type_ids"]

        label_dict = dict()
        batch_start_position_labels = list()
        batch_end_position_labels = list()
        batch_match_positions = list()

        batch_metadata = list()

        # start, end position 处理偏移
        for instance in instances:

            query_offset = 1 + len(instance["query"]) + 1  # CLS + query + SEP
            start_positions = instance.get("start_positions", None)
            end_positions = instance.get("end_positions", None)

            metadata = {"query": instance["query"],
                        "context": instance["context"]}

            batch_metadata.append(metadata)

            if start_positions is not None and end_positions is not None:
                metadata["positions"] = zip(start_positions, end_positions)

                start_positions = [(query_offset + start_position) for start_position in start_positions]
                start_position_labels = torch.zeros(batch_max_length)

                for start_position in start_positions:
                    if start_position < batch_max_length - 1:
                        start_position_labels[start_position] = 1

                batch_start_position_labels.append(start_position_labels)

                end_positions = [(query_offset + end_position) for end_position in end_positions]
                end_position_labels = torch.zeros(batch_max_length)

                for end_position in end_positions:

                    if end_position < batch_max_length - 1:
                        end_position_labels[end_position] = 1

                batch_end_position_labels.append(torch.tensor(end_position_labels, dtype=torch.long))

                # match position
                match_positions = torch.zeros(size=(batch_max_length, batch_max_length))

                for start_position, end_position in zip(start_positions, end_positions):

                    if start_position < batch_max_length - 1 and end_position < batch_max_length - 1:
                        match_positions[start_position, end_position] = 1

                batch_match_positions.append(match_positions)

        batch_start_position_labels = torch.stack(batch_start_position_labels)
        batch_end_position_labels = torch.stack(batch_end_position_labels)
        batch_match_positions = torch.stack(batch_match_positions)

        label_dict["start_position_labels"] = batch_start_position_labels
        label_dict["end_position_labels"] = batch_end_position_labels

        label_dict["match_position_labels"] = batch_match_positions

        return MRCModelInputs(batch_size=batch_size,
                              model_inputs={"input_ids": batch_inputs["input_ids"],
                                            "attention_mask": batch_inputs["attention_mask"],
                                            "token_type_ids": batch_inputs["token_type_ids"],
                                            "sequence_mask": batch_sequence_mask,
                                            "metadata": batch_metadata},
                              labels=label_dict)
