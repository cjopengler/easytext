#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
bio 的 schema 转换

Authors: PanXu
Date:    2021/02/03 09:39:00
"""

from typing import List


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


def bmes_to_bio(sequence_labels: List[str]) -> List[str]:
    """
    将 BMES schema 转换成 BIO schema
    :param sequence_labels: BMES schema 的 label 序列
    :return: BIO schema 的 label 序列
    """
    bio_sequence_labels = list()

    for label in sequence_labels:

        if label[0] == "B":
            bio_label = "B" + label[1:]
        elif label[0] == "M":
            bio_label = "I" + label[1:]
        elif label[0] == "E":
            bio_label = "I" + label[1:]
        elif label[0] == "S":
            bio_label = "B" + label[1:]
        elif label[0] == "O":
            bio_label = "O" + label[1:]
        else:
            raise RuntimeError(f"非法的 BMES label: {label}")

        bio_sequence_labels.append(bio_label)

    return bio_sequence_labels
