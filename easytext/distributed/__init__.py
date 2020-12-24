#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
分布式训练

Authors: PanXu
Date:    2020/11/29 17:52:00
"""

from .distributed import Distributed
from .synchronized import Synchronized
from .parameter import ProcessGroupParameter, DistributedDataParallelParameter
