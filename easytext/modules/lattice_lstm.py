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
from typing import Tuple

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


class LatticeLstm(Module):
    """
    Lattice Lstm
    """
    pass
