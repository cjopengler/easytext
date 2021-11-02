#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2021 PanXu, Inc. All Rights Reserved
#
"""
brief

Authors: PanXu
Date:    2021/10/27 13:38:00
"""
import torch
from easytext.model import ModelOutputs


class MRCNerOutput(ModelOutputs):
    """

    """

    def __init__(self,
                 start_logits: torch.Tensor,
                 end_logits: torch.Tensor,
                 match_logits: torch.Tensor,
                 mask: torch.Tensor):
        super().__init__(logits=None)

        self.start_logits = start_logits
        self.end_logits = end_logits
        self.match_logits = match_logits
        self.mask = mask
