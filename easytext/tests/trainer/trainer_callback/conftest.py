#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
brief

Authors: PanXu
Date:    2020/10/16 16:40:00
"""
import os
import pytest

from torch.utils.tensorboard import SummaryWriter

from easytext.tests import ROOT_PATH


@pytest.fixture(scope="package")
def summary_writer():
    log_dir = "data/tensorboard"
    log_dir = os.path.join(ROOT_PATH, log_dir)

    return SummaryWriter(log_dir=log_dir)
