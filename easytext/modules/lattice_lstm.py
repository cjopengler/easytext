#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
基于 <<Chinese NER Using Lattice LSTM>> 论文
论文地址: https://www.aclweb.org/anthology/P18-1144/

实现 lattice lstm 模型
相关说明文档参考:

docs/ner/Chinese NER Using Lattice LSTM.md

Authors: PanXu
Date:    2021/01/20 19:48:00
"""
import logging
from typing import Tuple, List, Dict

import numpy as np

import torch
from torch import nn
from torch.nn import init
from torch.nn import Module, Parameter


class WordLSTMCell(Module):
    """
    word cell

    相关说明文档: docs/ner/Chinese NER Using Lattice LSTM.md

    Part1 中，计算的结果，运算得到 c^w_{b,e}
    """

    def __init__(self, input_size, hidden_size, bias=True):
        """
        初始化
        :param input_size: w_{b,e} 也就是词向量的维度
        :param hidden_size: 输出的隐层维度, 注意实际会 hidden_size*3
        :param bias: 是否有 bias
        """

        super(WordLSTMCell, self).__init__()

        # input size
        self.input_size = input_size

        # 输出的 hidden size
        self.hidden_size = hidden_size

        # 是否使用 bias
        self.use_bias = bias

        # W*[x^w_{b,e}; h^c_b] + b = weight_ih*(x^w_{b,e}) + weight_hh*(h^c_b) + b 计算过程
        # weight_ih*(x^w_{b,e}) 计算该部分的参数
        # 注意: 3 * hidden_size 是因为一次性将 i, f, o 三个值计算出来，再通过 split 分开得到 3 个值，所以需要乘以 3
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))

        # weight_hh*(h^c_b) 计算该部分的参数
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset 参数
        :return:
        """
        init.orthogonal(self.weight_ih)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)

        with torch.no_grad():
            self.weight_hh.copy_(weight_hh_data)

        if self.bias is not None:
            init.constant(self.bias, val=0)

    def forward(self, input_, hx) -> torch.Tensor:
        """
        Args:
            input_: 是词向量，也就是 x^w_{b,e} 的词向量， size: (B, input_size)
            hx: (h_0, c_0), 是 h^c_b, 也就是在 b 处的隐层输出向量，size: (B, hidden_size).
        Returns:
            c_1: 是 part1 部分计算的结果，是 c^w_{b,e} 该值, size: (B, hidden_size)
        """

        # h_0: h^c_b 也就是 [b,e] 的 b 所在的 h
        h_0, c_0 = hx
        batch_size = h_0.size(0)

        if self.bias is not None:
            bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # weight_hh * h^c_b + b

        if self.bias is not None:
            wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        else:
            wh_b = torch.mm(h_0, self.weight_hh)

        # weight_ih * x^w_{b,e}
        wi = torch.mm(input_, self.weight_ih)

        # 计算 f, i, g
        f, i, g = torch.split(wh_b + wi, split_size_or_sections=self.hidden_size, dim=1)

        # 最终计算出 c^w_{b,e}
        c_1 = torch.sigmoid(f)*c_0 + torch.sigmoid(i)*torch.tanh(g)
        return c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class MultiInputLSTMCell(nn.Module):

    """
    结合 WordLSTMCell 和 LSTM Cell 的计算.

    Part2 和 Part3 计算, 计算得到多个 i, 并将 word (WordLSTMCell 计算的结果) 与 char 的 c_t 合并在一起。
    """

    def __init__(self, input_size, hidden_size, bias=True):
        """
        初始化
        :param input_size: input_size, 是指输入的 字 的 embedding 维度，也就是 x^c_j
        :param hidden_size: 输出的隐层维度
        :param bias: True: 使用 bias; False: 不使用 bias
        """

        super().__init__()

        # 输入的 embedding 维度
        self.input_size = input_size

        # 输出的 隐层 embedding 维度
        self.hidden_size = hidden_size

        # 与 x^c_j 进行相乘的参数
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))

        # 与 h^c_{j-1} 进行相乘的参数
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))

        # 用来计算 x^e_{b,e}(WordLSTMCell 计算出的结果) 与 当前 x^c_j 的系数的参数, 这是与 x^c_j 乘积的部分
        self.alpha_weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, hidden_size))

        # 用来计算 x^e_{b,e}(WordLSTMCell 计算出的结果) 与 当前 x^c_j 的系数的参数, 这是与 x^e_{b,e} 乘积的部分
        self.alpha_weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, hidden_size))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
            self.alpha_bias = nn.Parameter(torch.FloatTensor(hidden_size))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('alpha_bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        reset 参数
        :return:
        """
        init.orthogonal(self.weight_ih)
        init.orthogonal(self.alpha_weight_ih)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)

        with torch.no_grad():
            self.weight_hh.copy_(weight_hh_data)

        alpha_weight_hh_data = torch.eye(self.hidden_size)
        alpha_weight_hh_data = alpha_weight_hh_data.repeat(1, 1)

        with torch.no_grad():
            self.alpha_weight_hh.copy_(alpha_weight_hh_data)

        # The bias is just set to zero vectors.
        if self.bias is not None:
            init.constant(self.bias, val=0)
            init.constant(self.alpha_bias, val=0)

    def forward(self, input_, c_input, hx) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        注意: 当前模型仅仅支持 batch_size = 1
        :param input_: 字 embedding 输入向量
        :param c_input: 由 WordLSTMCell 计算得出的 所有 词的向量。
        :param hx: 在 j-1 步的输出, h_{j-1}, c_{j-1}
        :return: h_{j}, c_{j} 当前 cell 输出的隐层 和 cell 输出
        """

        # h^c_{j-1}, c^c_{j-1}, 前一个输出的 h, c, 该函数运算 lstm 一个 cell, 并得到 h^c_j, c^c_j
        h_0, c_0 = hx

        batch_size = h_0.size(0)

        # 注意只能处理 batch_size 为 1 的情况
        assert(batch_size == 1)

        if self.bias is not None:
            bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # $W*[x^c_j;h^c_{j-1}] + b = (weight_hh * h^c_{j-1}) + (weight_ih * x^c_j) + b$
        # $(weight_hh * h^c_{j-1}) + b$
        if self.bias is not None:
            wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        else:
            wh_b = torch.mm(h_0, self.weight_hh)

        # weight_ih * x^c_j
        wi = torch.mm(input_, self.weight_ih)
        # 计算 i, o, g, g 就是 $\tilde{c_j}$
        i, o, g = torch.split(wh_b + wi, split_size_or_sections=self.hidden_size, dim=1)

        # 计算 i, o, g
        i = torch.sigmoid(i)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        # c_sum, 是指当前j为结尾，一共命中了多少词
        c_num = len(c_input)

        if c_num == 0:
            # 没有命中词，则使用常规的 lstm 方法进行处理
            f = 1 - i
            c_1 = f*c_0 + i*g
            h_1 = o * torch.tanh(c_1)
        else:
            # 命中了多个词
            # 将所有命中词的向量，组合成一个向量, 按照0维合并
            c_input_var = torch.cat(c_input, 0)

            # 缩减维度，去掉 batch_size
            c_input_var = c_input_var.squeeze(1)

            # 计算 part3 中的 $i^c_{b,e}$
            # i^c_{b,e} = W*[x^c_j;c^w_{b,e}] + b = alpha_weight_ih*x^c_j + alpha_weight_hh * c^w_{b,e}
            # 其中 c^w_{b,e} 是在 WordLSTMCell 中计算的结果

            if self.alpha_bias is not None:
                alpha_wi = torch.addmm(self.alpha_bias, input_, self.alpha_weight_ih).expand(c_num, self.hidden_size)
            else:
                alpha_wi = torch.mm(input_, self.alpha_weight_ih).expand(c_num, self.hidden_size)

            alpha_wh = torch.mm(c_input_var, self.alpha_weight_hh)

            # alpha 就是 i^c_{b,e}
            alpha = torch.sigmoid(alpha_wi + alpha_wh)

            # 将所有的 i^c_j 与 所有的 i^c_{b,e} 组合在一起, 进行 softmax 计算
            alpha = torch.exp(torch.cat([i, alpha],0))
            alpha_sum = alpha.sum(0)
            alpha = torch.div(alpha, alpha_sum)

            # 最后一步，将 g=$\tilde{c_j}$, 以及 所有 c^w_{b,e} 放在一起，分别乘以权重进行计算
            merge_i_c = torch.cat([g, c_input_var],0)

            # 分别乘以权重，得到 c_1
            c_1 = merge_i_c * alpha
            c_1 = c_1.sum(0).unsqueeze(0)

            # 与常规 lstm 一样计算 h_1
            h_1 = o * torch.tanh(c_1)
        return h_1, c_1

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class LatticeLSTM(nn.Module):
    """
    基于 MultiInputLSTMCell 的 LSTM 模型
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 gaz_word_embedding_dim: int,
                 gaz_word_embedding: torch.Tensor,
                 gaz_word_embedding_dropout: float,
                 left2right: bool):
        """
        Lattice 初始出啊
        :param input_dim: 输入的维度
        :param hidden_dim: 隐层输出的维度
        :param gaz_word_embedding_dim: gaz 词向量的维度
        :param gaz_word_embedding: gaz 的词向量
        :param gaz_word_embedding_dropout: gaz 词向量的 dropout
        :param left2right: 从左向右 还是 从右向左，就是语言模型的两个方向
        """

        super().__init__()

        self.hidden_dim = hidden_dim
        self.word_emb = gaz_word_embedding

        self.word_dropout = nn.Dropout(gaz_word_embedding_dropout)

        self.rnn = MultiInputLSTMCell(input_dim, hidden_dim)
        self.word_rnn = WordLSTMCell(gaz_word_embedding_dim, hidden_dim)

        self.left2right = left2right

        self.reset_parameters()

    def reset_parameters(self):
        pass

    def forward(self,
                input: torch.Tensor,
                skip_input: List[List[List[List]]],
                hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行模型
        :param input: 字序列向量, shape: (B, seq_len, embedding_size),
                      但是 batch_size = 1， 必须是 1，也就是不支持其他 batch_size
        :param skip_input: skip_input: 是4维 list, 因为 B=1, 所以内部是一个 3维list,
                           该 3维 list 总长度是 seq_len(与字符序列是一样长度的).
                           每一个元素，是对应到词汇表中词的 id 和 对应的长度。例如:
                           [[], [[25,13],[2,4]], [], [[33], [2]], []], 表示在字序列中，第 2个 字，所对应的词 id 是 25 和13 , 对应的长度是 2 和 4。
                           例如: "到 长 江 大 桥", 该序列长度是 5， 所以 skip_input 也是 5, 其中 "长" index=1,
                           对应 "长江" 和 "长江大桥", 其中 "长江" 在词汇表中的 id 是25, 长度是 2;
                           "长江大桥" 对应词汇表中 id 是 13， 长度是 4;
                            同样 "大桥", 对应 词汇表 id 33, 长度是 2.
        :param hidden: 预定义的 (h,c) 输入
        :return: (h1, c1), ..., (hn, cn), 返回的是 sequence 隐层序列, hj 或 cj 的 shape: (B, seq_len, hidden_dim), 其中 B=1
        """

        input = input.transpose(1, 0)

        seq_len = input.size(0)
        batch_size = input.size(1)

        # 只能处理 batch_size 为 1
        assert (batch_size == 1)

        # 因为 batch_size == 1, 所以仅仅取第一个
        skip_input = skip_input[0]

        if not self.left2right:  # 如果是 right2left 需要将 构成的词汇表也进行逆向
            skip_input = convert_forward_gaz_to_backward(skip_input)

        hidden_out = []  # h
        memory_out = []  # c

        if hidden:  # 如果 hidden_{t-1} 存在，则使用; 否则，使用 0
            (hx, cx) = hidden
        else:
            hx = torch.zeros(batch_size, self.hidden_dim, device=input.device)
            cx = torch.zeros(batch_size, self.hidden_dim, device=input.device)

        id_list = range(seq_len)

        if not self.left2right:
            id_list = list(reversed(id_list))

        # 用来存储 WordLSTMCell 计算得到的 c^w_t, 就是在某个位置上 word 的 ct
        # 初始是空的，会通过 skip_input， 在计算的过程中逐渐填充
        # 注意: 当 t=0 时, input_c_list[0] 一定是 空的。因为，第一个没有前面的字，所以无法组成词。
        input_c_list = init_list_of_objects(seq_len)

        for t in id_list:

            (hx, cx) = self.rnn(input[t], input_c_list[t], (hx, cx))

            hidden_out.append(hx)
            memory_out.append(cx)

            if skip_input[t]:  # 如果当前 t 位置的字，有词构成，那么，则使用 WordLSTMCell 生成 c^w_t

                # 一共匹配了 多少个 词
                matched_num = len(skip_input[t][0])

                # 获取所有 word id, 组成 word id tensor, 注意是 多个 word id, 不是一个
                word_var = torch.tensor(skip_input[t][0], device=input.device, dtype=torch.long)

                # 获取所有 word id 以及词向量
                word_emb = self.word_emb(word_var)
                word_emb = self.word_dropout(word_emb)

                # 计算所有 word id 的 c^w_t, 一次性计算得到多个 word id 的 c^w
                ct = self.word_rnn(word_emb, (hx, cx))

                assert (ct.size(0) == len(skip_input[t][1]))

                # 将计算得到的所有 c^w_t 全部存放到 input_c_list 对应的 字 的位置上。
                for idx in range(matched_num):

                    length = skip_input[t][1][idx]

                    if self.left2right:
                        # if t+length <= seq_len -1:
                        input_c_list[t + length - 1].append(ct[idx, :].unsqueeze(0))
                    else:
                        # if t-length >=0:
                        input_c_list[t - length + 1].append(ct[idx, :].unsqueeze(0))

        if not self.left2right:
            hidden_out = list(reversed(hidden_out))
            memory_out = list(reversed(memory_out))

        output_hidden, output_memory = torch.cat(hidden_out, 0), torch.cat(memory_out, 0)

        # 输出的 shape: (batch, seq_len, hidden_dim)
        return output_hidden.unsqueeze(0), output_memory.unsqueeze(0)


def init_list_of_objects(size) -> List[List]:
    """
    构建一个 word ct 的 list
    :param size: sequence 的 长度
    :return: 二维 list, 因为每一个 字 的位置，可能有多个词，所以有多个 wrod ct
    """

    list_of_objects = list()
    for i in range(0, size):
        list_of_objects.append(list())

    return list_of_objects


def convert_forward_gaz_to_backward(forward_gaz):
    # print forward_gaz
    length = len(forward_gaz)
    backward_gaz = init_list_of_objects(length)
    for idx in range(length):
        if forward_gaz[idx]:
            assert (len(forward_gaz[idx]) == 2)
            num = len(forward_gaz[idx][0])
            for idy in range(num):
                the_id = forward_gaz[idx][0][idy]
                the_length = forward_gaz[idx][1][idy]
                new_pos = idx + the_length - 1
                if backward_gaz[new_pos]:
                    backward_gaz[new_pos][0].append(the_id)
                    backward_gaz[new_pos][1].append(the_length)
                else:
                    backward_gaz[new_pos] = [[the_id], [the_length]]
    return backward_gaz

