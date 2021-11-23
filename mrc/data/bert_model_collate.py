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

        batch_text_pairs = [(instance["query"], instance["context"]) for instance in instances]

        batch_inputs = self._tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_text_pairs,
                                                         truncation=True,
                                                         padding=True,
                                                         max_length=self._max_length,
                                                         return_length=True,
                                                         add_special_tokens=True,
                                                         return_special_tokens_mask=True,
                                                         return_offsets_mapping=True,
                                                         # return_token_type_ids=True,
                                                         return_tensors="pt")

        batch_token_ids = batch_inputs["input_ids"]

        batch_token_type_ids = batch_inputs["token_type_ids"]
        batch_max_len = max(batch_inputs["length"])

        batch_offset_mapping = batch_inputs["offset_mapping"]
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
        for instance, token_ids, token_type_ids, offset_mapping in zip(instances,
                                                                       batch_token_ids,
                                                                       batch_token_type_ids,
                                                                       batch_offset_mapping):

            start_positions = instance.get("start_positions", None)
            end_positions = instance.get("end_positions", None)

            metadata = {"query": instance["query"],
                        "context": instance["context"]}

            batch_metadata.append(metadata)

            if start_positions is not None and end_positions is not None:

                # 是因为在 offset 中, 对于 index 的设置，就是 [start, end)
                end_positions = [end_pos + 1 for end_pos in end_positions]

                # 因为 query 和 context 拼接在一起了，所以 start_position 和 end_position 的位置要重新映射
                origin_offset2token_idx_start = {}
                origin_offset2token_idx_end = {}

                last_token_start = 0
                last_token_end = 0

                for token_idx in range(len(token_ids)):
                    # query 的需要过滤
                    if token_type_ids[token_idx] == 0:
                        continue

                    # 获取每一个 token_start 和 end
                    token_start, token_end = offset_mapping[token_idx]
                    token_start = token_start.item()
                    token_end = token_end.item()

                    # skip [CLS] or [SEP], offset 中 (0, 0) 表示的就是 CLS 或者 SEP
                    if token_start == token_end == 0:
                        continue

                    # 保存下最后的 start 和 end
                    last_token_start = token_start
                    last_token_end = token_end

                    # token_start 对应的就是 context 中的实际位置，与 start_position 与 end_position 是对应的
                    # token_idx 是 query 和 context 拼接在一起后的 index，所以 这就是 start_position 映射后的位置
                    origin_offset2token_idx_start[token_start] = token_idx
                    origin_offset2token_idx_end[token_end] = token_idx

                # 将原始数据中的  start_positions 映射到 拼接 query context 之后的位置
                new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions
                                       if start <= last_token_start]
                new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions
                                     if end <= last_token_end]

                metadata["positions"] = zip(start_positions, end_positions)

                start_position_labels = torch.zeros(batch_max_len, dtype=torch.long)

                for start_position in new_start_positions:
                    if start_position < batch_max_len - 1:
                        start_position_labels[start_position] = 1

                batch_start_position_labels.append(start_position_labels)

                end_position_labels = torch.zeros(batch_max_len, dtype=torch.long)

                for end_position in new_end_positions:

                    if end_position < batch_max_len - 1:
                        end_position_labels[end_position] = 1

                batch_end_position_labels.append(end_position_labels)

                # match position
                match_positions = torch.zeros(size=(batch_max_len, batch_max_len), dtype=torch.long)

                for start_position, end_position in zip(new_start_positions, new_end_positions):

                    if start_position < batch_max_len - 1 and end_position < batch_max_len - 1:
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
