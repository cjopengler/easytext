#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
接口: 将数据转换成一个 instance, 也就是一个训练的样本

Authors: PanXu
Date:    2021/08/19 08:54:00
"""


class InstanceFactory:
    """
    接口: 将数据转换成一个 instance, 也就是一个训练的样本
    训练时: 在构建数据集的时候，会构建 instance
    预测时: 在预测的时候，也会使用同样的方法来构建 instance
    这样做的收益在于，保证了训练和预测时候使用了同样的数据处理方式。
    """

    def create_instance(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
