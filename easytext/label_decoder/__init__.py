#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
label decoder

Authors: PanXu
Date:    2020/07/05 10:19:00
"""

from .model_label_decoder import ModelLabelDecoder

from .label_decoder.label_decoder import LabelDecoder
from .label_decoder.sequence_label_decoder import SequenceLabelDecoder

from .label_index_decoder.label_index_decoder import LabelIndexDecoder
from .label_index_decoder.crf_label_index_decoder import CRFLabelIndexDecoder
from .label_index_decoder.sequence_max_label_index_decoder import SequenceMaxLabelIndexDecoder
from .label_index_decoder.max_label_index_decoder import MaxLabelIndexDecoder