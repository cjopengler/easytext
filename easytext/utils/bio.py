#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
序列标注用的 bioul

Authors: panxu(panxu@baidu.com)
Date:    2020/05/13 17:07:00
"""

import logging
from typing import List, Dict, Tuple

import torch
from easytext.data import LabelVocabulary


def fill(sequence_label: List[str], begin_index: int, end_index: int, tag: str) -> "None":
    """
    对 sequence label 填充 B I O
    :param sequence_label: 需要填充的序列
    :param begin_index: 开始位置
    :param end_index: 结束位置 [begin_index, end_index)
    :param tag: 对 sequence_label 中填充的 "B-tag", "I-tag"
    :return:
    """
    if len(sequence_label) == 0:
        raise RuntimeError("sequence_label size is 0")

    sequence_label[begin_index:end_index] = [f"I-{tag}"] * (end_index - begin_index)
    sequence_label[begin_index] = f"B-{tag}"


def decode_one_sequence_logits_to_label(sequence_logits: torch.Tensor,
                                        vocabulary: LabelVocabulary) -> Tuple[List[str], List[int]]:
    """
    对 输出 sequence logits 进行解码, 是仅仅一个 sequence 进行解码，而不是 batch sequence 进行解码。
    batch sequence 解码需要进行循环
    :param sequence_logits: shape: (seq_len, label_num),
    是 mask 之后的有效 sequence，而不是包含 mask 的 sequecne logits.
    :return: sequence label, B, I, O 的list 以及 label 对应的 index list
    """

    if len(sequence_logits.shape) != 2:
        raise RuntimeError(f"sequence_logits shape 是 (seq_len, label_num)， 现在是 {sequence_logits.shape}")

    idel_state, span_state = 0, 1

    sequence_length = sequence_logits.size(0)

    state = idel_state

    # 按权重进行排序 indices shape: (seq_len, label_num)
    sorted_sequence_indices = torch.argsort(sequence_logits, dim=-1, descending=True)

    sequence_label = list()
    sequence_label_indices = list()

    for i in range(sequence_length):

        indices = sorted_sequence_indices[i, :].tolist()

        if state == idel_state:

            # 循环寻找，直到找到一个合理的标签
            for index in indices:

                label = vocabulary.token(index)

                if label[0] == "O":

                    sequence_label.append(label)
                    sequence_label_indices.append(index)
                    state = idel_state
                    break
                elif label[0] == "B":
                    sequence_label.append(label)
                    sequence_label_indices.append(index)
                    state = span_state
                    break
                else:
                    # 其他情况 "I" 这是不合理的，所以这个逻辑是找到一个合理的标签
                    pass

        elif state == span_state:
            for index in indices:

                label = vocabulary.token(index)

                if label[0] == "B":
                    sequence_label.append(label)
                    sequence_label_indices.append(index)
                    state = span_state
                    break
                elif label[0] == "O":
                    sequence_label.append(label)
                    sequence_label_indices.append(index)
                    state = idel_state
                    break
                elif label[0] == "I":
                    sequence_label.append(label)
                    sequence_label_indices.append(index)
                    state = span_state
                    break
                else:
                    raise RuntimeError(f"{label} 不符合 BIO 格式")
        else:
            raise RuntimeError(f"state is error: {state}")

    return sequence_label, sequence_label_indices


def decode_one_sequence_label_to_span(sequence_label: List[str]) -> List[Dict]:
    """
    对 BIO 序列进行解码成 List. 例如:

    ["B-Per", "I-Per", "O", "B-Loc"] ->
    [ {"label": "Per", "begin": 0, "end": 2},
      {"label": "Loc", "begin": 3, "end": 4} ]
    :param sequence_label: BIO 序列。
    :return: 解码好的字典列表
    """
    idel_state, span_state = 0, 1

    spans = list()
    begin = None
    tag = None

    state = idel_state
    for i, label in enumerate(sequence_label):

        if state == idel_state:
            if label[0] == "B":
                begin = i
                tag = label[2:]
                state = span_state
            elif label[0] == "O":
                pass
            elif label[0] == "I":
                logging.warning(f"{sequence_label} 有不满足 BIO 格式的问题")
            else:
                raise RuntimeError(f"{label} schema 不符合 BIO")

        elif state == span_state:
            if label[0] == "B":
                span = {"label": tag,
                        "begin": begin,
                        "end": i}
                spans.append(span)
                begin = i
                tag = label[2:]
                state = span_state
            elif label[0] == "O":
                span = {"label": tag,
                        "begin": begin,
                        "end": i}
                spans.append(span)
                begin = None
                tag = None
                state = idel_state
            elif label[0] == "I":
                state = span_state
            else:
                raise RuntimeError(f"{label} schema 不符合 BIO")
        else:
            raise RuntimeError(f"{state} 错误，应该是 在 [{idel_state}, {span_state}] ")

    if state == span_state:
        span = {"label": tag,
                "begin": begin,
                "end": len(sequence_label)}
        spans.append(span)

    return spans


def decode(batch_sequence_logits: torch.Tensor,
           mask: torch.LongTensor,
           vocabulary: LabelVocabulary) -> List[List[Dict]]:
    """
    对 sequence logits 进行 BIO 解码，得到 span.

    例如:
    batch_sequence_logits:
    [[[0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.5, 0.2, 0.3]],
     [[0.1, 0.7, 0.2], [0.2, 0.2, 0.6], [0.5, 0.2, 0.3]]]

    返回结果:
    [[{"label": "Tag", "begin": 0, "end": 1}],
     [{"label": "Tag", "begin": 1, "end": 3}]]

    返回:
    :param batch_sequence_logits: shape: (B, seq_len, label_num) 经过模型预测后得到每一个label的概率分布
    :param mask: shape: (B, seq_len) mask
    :param vocabulary: label的 vocabulary
    :return: decode 之后的 span 字典列表
    """

    if batch_sequence_logits.dim() != 3:
        raise RuntimeError(f"batch_sequence_logits shape 错误, 应该是 (B, seq_len, label_num), "
                           f"而现在是 {batch_sequence_logits.shape}")

    if (mask is not None) and (mask.dim() != 2):
        raise RuntimeError(f"mask shape 错误, 应该是 (B, seq_len), "
                           f"而现在是 {mask.shape}")

    batch = batch_sequence_logits.size(0)

    # mask shape: (B, seq_len)
    if mask is None:
        mask = torch.ones(size=(batch_sequence_logits.shape[0], batch_sequence_logits.shape[1]),
                          dtype=torch.long)

    sequence_length = mask.sum(dim=-1).tolist()

    spans = list()
    for i in range(batch):
        sequence_label, _ = decode_one_sequence_logits_to_label(
            sequence_logits=batch_sequence_logits[i, :sequence_length[i]],
            vocabulary=vocabulary)
        sequence_span = decode_one_sequence_label_to_span(sequence_label)
        spans.append(sequence_span)

    return spans


def decode_label_index_to_span(batch_sequence_label_index: torch.Tensor,
                               mask: torch.ByteTensor,
                               vocabulary: LabelVocabulary) -> List[List[Dict]]:
    """
    将 label index 解码 成span

    batch_sequence_label shape:(B, seq_len)  (B-T: 0, I-T: 1, O: 2)
    [[0, 1, 2],
     [2, 0, 1]]

     对应label序列是:
     [[B, I, O],
      [O, B, I]]

     解码成:

     [[{"label": T, "begin": 0, "end": 2}],
      [{"label": T, "begin": 1, "end": 3}]]

    :param batch_sequence_label_index: shape: (B, seq_len), label index 序列
    :param mask: 对 batch_sequence_label 的 mask
    :param vocabulary: label 词汇表
    :return: 解析好的span列表
    """

    spans = list()

    if mask is None:
        mask = torch.ones(size=(batch_sequence_label_index.shape[0], batch_sequence_label_index.shape[1]),
                          dtype=torch.long)
    mask = mask.bool()
    for sequence_label_index, mask1d in zip(batch_sequence_label_index, mask):
        label_indices = torch.masked_select(sequence_label_index, mask=mask1d).tolist()

        sequence_label = [vocabulary.token(index) for index in label_indices]

        span = decode_one_sequence_label_to_span(sequence_label=sequence_label)
        spans.append(span)

    return spans


def span_intersection(span_list1: List[Dict],
                      span_list2: List[Dict]) -> List[Dict]:
    """
    对 两个 span 列表 求交集
    :param span_list1:
    :param span_list2:
    :return: 交集的结果
    """

    span_set1 = {(span["label"], span["begin"], span["end"]) for span in span_list1}
    span_set2 = {(span["label"], span["begin"], span["end"]) for span in span_list2}

    intersection = [{"label": span[0], "begin": span[1], "end": span[2]} for span in span_set1 & span_set2]

    return intersection


def ibo1_to_bio(sequence_label: List[str]) -> List[str]:
    """
    IBO1 的序列标签格式转换成 bio 格式。ibo1格式是指 "I-Label" 作为一个序列的开始, 直到遇到 O。
    "B-Label" 是指连续 "I-Label" 是多个 span 的时候，除了第一个span，其他都是 "B-Label"。
    另外 BIO 的格式也兼容，比如 "B-Lable", "I-Label" 这种BIO格式也是正确的。
    例子如下:
    "I-Label1 I-Label1 O I-Label1 I-Label1 I-Label2 I-Label2 O I-Label1 I-Label1 B-Label1 I-Label1 O B-Label I-Label"

    这里包含的spans:
    "[I-Label1 I-Label1] O [I-Label1 I-Label1] [I-Label2 I-Label2] O [I-Label1 I-Label1] [B-Label1 I-Label1] O [B-Label I-Label"]

    :param sequence_label:
    :return:
    """

    bio: List[str] = list()

    idel_state = 0
    span_state = 1
    state = idel_state

    for i, label in enumerate(sequence_label):

        if state == idel_state:
            if label == "O":
                state = idel_state
                bio.append(label)
            elif label[0] == "I":
                state = span_state
                # 将 I-label 转换成 B-label
                b_label = "B" + label[1:]
                bio.append(b_label)
            elif label[0] == "B":
                # 这种情况是 BIO 标注，认为是对的
                state = span_state
                bio.append(label)
        elif state == span_state:
            if label == "O":
                state = idel_state
                bio.append(label)
            elif label[0] == "I":
                if bio[-1][1:] == label[1:]:
                    # 与先前的相同，也就是 I-label1 I-label1 的情况，所以直接append即可
                    state = span_state
                    bio.append(label)
                else:
                    # 与先前不一样说明是新的，也就是 I-label1 I-label2 的情况
                    state = span_state
                    b_label = "B" + label[1:]
                    bio.append(b_label)
            elif label[0] == "B":
                # 这是新的了 也就是 I-label1 B-label1 情况
                state = span_state
                bio.append(label)
        else:
            raise RuntimeError(f"非法的状态: {state}")

    return bio


def allowed_transitions(label_vocabulary: LabelVocabulary) -> List[Tuple[int, int]]:
    """
    给定 label 字典，计算 IBO schema 下的 转移矩阵 mask. 因为在 BIO 下面，比如 ["O", "I-Per"]
    或者 ["B-Per", "I-Loc"], 这些转移是不被允许的。该函数，计算mask. 对于允许的转移返回
    pair, (from_label_id, to_label_id)

    这里要特别注意的是: 返回的 allowed pair, 是带有 START 和 STOP的。他们的 index 分别是:
    label_vocabulary.label_size, label_vocabulary.label_size+1

    注意 label_vocabulary.padding_index 也是 label_vocabulary.label_size。虽然是一样的，
    但是因为 START 是用在转移矩阵这个特定场景下，所以不会产生冲突。

    :param label_vocabulary: label 词汇表
    :return: 所有被允许转移的 (from_label_id, to_label_id) pair 对列表。
    """

    num_labels = label_vocabulary.label_size
    start_index = num_labels
    end_index = num_labels + 1

    labels = [(i, label_vocabulary.token(i)) for i in range(label_vocabulary.label_size)]
    labels_with_boundaries = labels + [(start_index, "START"), (end_index, "END")]

    allowed = []
    for from_label_index, from_label in labels_with_boundaries:
        if from_label in ("START", "END"):
            from_tag = from_label
            from_entity = ""
        else:
            from_tag = from_label[0]
            from_entity = from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            if to_label in ("START", "END"):
                to_tag = to_label
                to_entity = ""
            else:
                to_tag = to_label[0]
                to_entity = to_label[1:]
            if _is_transition_allowed(from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def _is_transition_allowed(from_tag: str,
                           from_entity: str,
                           to_tag: str,
                           to_entity: str):
    """
    BIO 是否被允许转移。比如: "B-Per" "I-Per" 这是被允许的; 而 "B-Per" "I-Loc" 或者 "O", "I-Per" 这是不被允许的
    :param from_tag: The tag that the transition originates from. For example, if the
        label is ``I-PER``, the ``from_tag`` is ``I``.
    :param from_entity: The entity corresponding to the ``from_tag``. For example, if the
        label is ``I-PER``, the ``from_entity`` is ``PER``.
    :param to_tag: The tag that the transition leads to. For example, if the
        label is ``I-PER``, the ``to_tag`` is ``I``.
    :param to_entity: The entity corresponding to the ``to_tag``. For example, if the
        label is ``I-PER``, the ``to_entity`` is ``PER``.
    :return: True: 该转移是被允许的; False: 该转移是不被允许的。
    """

    if to_tag == "START" or from_tag == "END":
        # Cannot transition into START or from END
        return False

    if from_tag == "START":
        return to_tag in ('O', 'B')

    if to_tag == "END":
        return from_tag in ('O', 'B', 'I')

    return any([
        # Can always transition to O or B-x
        to_tag in ('O', 'B'),
        # Can only transition to I-x from B-x or I-x
        to_tag == 'I' and from_tag in ('B', 'I') and from_entity == to_entity
    ])
