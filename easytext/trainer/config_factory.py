#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
config 工厂

Authors: panxu(panxu@baidu.com)
Date:    2020/06/25 18:57:00
"""

from typing import Dict


class ConfigFactory:
    """
    config 创建工厂
    """

    def create(self) -> Dict:
        """
        创建 config
        """
        raise NotImplementedError()
