#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
#
"""
brief

Authors: panxu(panxu@baidu.com)
Date:    2020/06/01 15:40:00
"""
import logging
import math
from typing import Optional, List, Tuple

import torch
from torch import LongTensor, BoolTensor


def has_tensor(obj) -> bool:
    """
    检查 对象中是否包含 Tensor
    :param obj:
    :return: True: 包含 Tensor; False: 没有 Tensor
    """
    if isinstance(obj, torch.Tensor):
        return True
    elif isinstance(obj, dict):
        return any(has_tensor(value) for value in obj.values())
    elif isinstance(obj, (list, tuple)):
        return any(has_tensor(item) for item in obj)
    else:
        return False


def sequence_mask(sequence: LongTensor, padding_index: int = 0) -> BoolTensor:
    """
    计算 sequence 序列的 mask
    :param sequence: sequence index, 维度是 (B, SeqLen)
    :param padding_index: padding 的index, 一般是 0，也可以根据自己的padding index 来设置。
    :return: sequence mask, 注意是 BoolTensor, 根据需要可以转化成 Float 或者 Long
    """

    if sequence.dim() != 2:
        raise RuntimeError(f"Sequence 的维度 {sequence.dim()} 不是 2，也就是 (B, SeqLen)")

    return sequence != padding_index


def logsumexp(tensor: torch.Tensor,
              dim: int = -1,
              keepdim: bool = False) -> torch.Tensor:
    """
    `tensor.exp().sum(dim, keep=keepdim).log()`。 就是: log(exp(S01 + S02 + ...) + exp(S11 + S12 + ...))
    也被叫做 logadd 是 条件随机场中计算边缘概率用的公式。
    :param tensor: 要计算的 tensor
    :param dim: 沿着那个维度来进行 sum
    :param keepdim: 是否保持原先的维度, 默认是 False
    :return: logadd 值
    """

    # 计算 max_score, 是为了避免 tensor 中的数值太小或者太大而导致的溢出
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


def viterbi_decode(tag_sequence: torch.Tensor,
                   transition_matrix: torch.Tensor,
                   tag_observations: Optional[List[int]] = None,
                   allowed_start_transitions: torch.Tensor = None,
                   allowed_end_transitions: torch.Tensor = None) -> Tuple[List[int], torch.Tensor]:
    """
    viterbi 解码
    :param tag_sequence: tag sequence, shape (sequence_len, num_tag).
    :param transition_matrix: 转移矩阵, shape (num_tag, num_tag)
    :param tag_observations: 一般情况下这个参数是不会被设置的。A list of length ``sequence_length`` containing the class ids of observed
    elements in the sequence, with unobserved elements being set to -1. Note that
    it is possible to provide evidence which results in degenerate labelings if
    the sequences of tags you provide as evidence cannot transition between each
    other, or those transitions are extremely unlikely. In this situation we log a
    warning, but the responsibility for providing self-consistent evidence ultimately
    lies with the user.
    :param allowed_start_transitions: <BOS>, shape (num_tags,)，有额外增加 START 的时候对 转移矩阵的额外设置。
    :param allowed_end_transitions: <EOS>, shape (num_tag,)，有额外增加 STOP 的时候对转移矩阵的额外设置。
    :return: 1. viterbi_path : List[int] - viterbi 路径，也就是 max socre 路径。
             2. viterbi_score : torch.Tensor - viterbi 路径的分数, 就只有一个值。
    """

    sequence_length, num_tags = list(tag_sequence.size())

    has_start_end_restrictions = allowed_end_transitions is not None or allowed_start_transitions is not None

    if has_start_end_restrictions:

        if allowed_end_transitions is None:
            allowed_end_transitions = torch.zeros(num_tags)
        if allowed_start_transitions is None:
            allowed_start_transitions = torch.zeros(num_tags)

        num_tags = num_tags + 2
        new_transition_matrix = torch.zeros(num_tags, num_tags)
        new_transition_matrix[:-2, :-2] = transition_matrix

        # Start and end transitions are fully defined, but cannot transition between each other.
        # pylint: disable=not-callable
        allowed_start_transitions = torch.cat([allowed_start_transitions, torch.tensor([-math.inf, -math.inf])])
        allowed_end_transitions = torch.cat([allowed_end_transitions, torch.tensor([-math.inf, -math.inf])])
        # pylint: enable=not-callable

        # First define how we may transition FROM the start and end tags.
        new_transition_matrix[-2, :] = allowed_start_transitions
        # We cannot transition from the end tag to any tag.
        new_transition_matrix[-1, :] = -math.inf

        new_transition_matrix[:, -1] = allowed_end_transitions
        # We cannot transition to the start tag from any tag.
        new_transition_matrix[:, -2] = -math.inf

        transition_matrix = new_transition_matrix

    if tag_observations:
        if len(tag_observations) != sequence_length:
            raise RuntimeError("Observations were provided, but they were not the same length "
                                     "as the sequence. Found sequence of length: {} and evidence: {}"
                                     .format(sequence_length, tag_observations))
    else:
        tag_observations = [-1 for _ in range(sequence_length)]


    if has_start_end_restrictions:
        tag_observations = [num_tags - 2] + tag_observations + [num_tags - 1]
        zero_sentinel = torch.zeros(1, num_tags)
        extra_tags_sentinel = torch.ones(sequence_length, 2) * -math.inf
        tag_sequence = torch.cat([tag_sequence, extra_tags_sentinel], -1)
        tag_sequence = torch.cat([zero_sentinel, tag_sequence, zero_sentinel], 0)
        sequence_length = tag_sequence.size(0)

    path_scores = []
    path_indices = []

    if tag_observations[0] != -1:
        one_hot = torch.zeros(num_tags)
        one_hot[tag_observations[0]] = 100000.
        path_scores.append(one_hot)
    else:
        path_scores.append(tag_sequence[0, :])

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        summed_potentials = path_scores[timestep - 1].unsqueeze(-1) + transition_matrix
        scores, paths = torch.max(summed_potentials, 0)

        # If we have an observation for this timestep, use it
        # instead of the distribution over tags.
        observation = tag_observations[timestep]
        # Warn the user if they have passed
        # invalid/extremely unlikely evidence.
        if tag_observations[timestep - 1] != -1 and observation != -1:
            if transition_matrix[tag_observations[timestep - 1], observation] < -10000:
                logging.warning("The pairwise potential between tags you have passed as "
                               "observations is extremely unlikely. Double check your evidence "
                               "or transition potentials!")
        if observation != -1:
            one_hot = torch.zeros(num_tags)
            one_hot[observation] = 100000.
            path_scores.append(one_hot)
        else:
            path_scores.append(tag_sequence[timestep, :] + scores.squeeze())
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    viterbi_score, best_path = torch.max(path_scores[-1], 0)
    viterbi_path = [int(best_path.numpy())]
    for backward_timestep in reversed(path_indices):
        viterbi_path.append(int(backward_timestep[viterbi_path[-1]]))
    # Reverse the backward path.
    viterbi_path.reverse()

    if has_start_end_restrictions:
        viterbi_path = viterbi_path[1:-1]
    return viterbi_path, viterbi_score


def masked_softmax(vector: torch.FloatTensor,
                   mask: torch.ByteTensor):
    """
    计算带有 masked 的 softmax
    :param vector: shape: (B, seq_len)
    :param mask: shape: (B, seq_len),
    :return: (B, seq_len)
    """
    exp_vector = vector.exp()

    masked_vector = exp_vector * mask.float()

    return masked_vector / torch.sum(masked_vector, dim=-1, keepdim=True)



