#!/usr/bin/env python 3
# -*- coding: utf-8 -*-

#
# Copyright (c) 2020 PanXu, Inc. All Rights Reserved
#
"""
ner launcher

Authors: PanXu
Date:    2020/12/21 22:09:00
"""
from typing import Optional, Union, List
from tqdm import tqdm
import torch
import os
import logging
from typing import Dict, Union
import shutil
from argparse import ArgumentParser

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from transformers import BertTokenizer

from easytext.trainer import Trainer
from easytext.data import Vocabulary, LabelVocabulary, PretrainedVocabulary
from easytext.data import GloveLoader, SGNSLoader
from easytext.utils import log_util
from easytext.utils.seed_util import set_seed
from easytext.trainer import Launcher, Config
from easytext.distributed import ProcessGroupParameter
from easytext.utils.json_util import json2str
from easytext.component.register import Registry
from easytext.utils.bert_tokenizer import bert_tokenizer

from mrc.data import MSRAFlatNerDataset
from mrc.data import BertModelCollate
from mrc.models import MRCNer
from mrc.loss import MRCBCELoss
from mrc.metric import MrcModelMetricAdapter
from mrc.optimizer import MRCOptimizer
from mrc.optimizer import MRCLrScheduler


class MrcNerLauncher(Launcher):
    """
    mrc ner 训练的启动器
    """

    NEW_TRAIN = 0

    def __init__(self, config_file_path: str, train_type: int):
        config = Config(config_file_path=config_file_path, is_training=True)
        self.config = config
        super().__init__()
        self._train_type = train_type

    def _init_devices(self) -> Union[List[str], List[int]]:
        return self.config.devices

    def _init_process_group_parameter(self, rank: Optional[int]) -> Optional[ProcessGroupParameter]:

        if len(self._devices) > 1:
            return self.config.process_group_parameter
        else:
            return None

    def _preprocess(self):

        logging.info(f"config:\n{self.config}\n")

        serialize_dir = self.config.serialize_dir
        if self._train_type == MrcNerLauncher.NEW_TRAIN:
            # 清理 serialize dir
            if os.path.isdir(serialize_dir):
                shutil.rmtree(serialize_dir)
                os.makedirs(serialize_dir)

    def _start(self, rank: Optional[int], world_size: int, device: torch.device) -> None:

        is_distributed = rank is not None

        trainer = Trainer(serialize_dir=self.config.serialize_dir,
                          num_epoch=self.config.num_epoch,
                          model=self.config.model,
                          loss=self.config.loss,
                          metrics=self.config.metric,
                          optimizer_factory=self.config.optimizer,
                          lr_scheduler_factory=self.config.lr_scheduler,
                          grad_rescaled=self.config.grad_rescaled,
                          patient=self.config.patient,
                          num_check_point_keep=self.config.num_check_point_keep,
                          device=device,
                          is_distributed=is_distributed,
                          distributed_data_parallel_parameter=self.config.distributed_data_parallel_parameter)

        train_sampler = None

        if is_distributed:
            train_sampler = DistributedSampler(dataset=self.config.training_dataset, shuffle=False)

        train_data_loader = DataLoader(
            dataset=self.config.training_dataset,
            batch_size=self.config.train_batch_size,
            shuffle=(train_sampler is None),
            num_workers=0,
            collate_fn=self.config.model_collate,
            sampler=train_sampler
        )    

        validation_sampler = None

        if is_distributed:
            validation_sampler = DistributedSampler(dataset=self.config.training_dataset, shuffle=False)

        validation_data_loader = DataLoader(
            dataset=self.config.validation_dataset,
            batch_size=self.config.test_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.config.model_collate,
            sampler=validation_sampler
        )

        trainer.train(train_data_loader=train_data_loader,
                      validation_data_loader=validation_data_loader)


if __name__ == '__main__':
    set_seed()
    
    log_util.config(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--config", default="", help="训练配置文件")

    parsed_args = parser.parse_args()

    if parsed_args.config == "":
        logging.fatal("--config 参数为空!")
        exit(-1)
    logging.info(f"config file path: {parsed_args.config}")
    ner_launcher = MrcNerLauncher(config_file_path=parsed_args.config, train_type=MrcNerLauncher.NEW_TRAIN)
    ner_launcher()
