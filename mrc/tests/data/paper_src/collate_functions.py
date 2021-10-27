# encoding: utf-8

import torch
from typing import List


def collate_to_max_length(batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    pad to maximum length of this batch
    Args:
        batch: a batch of samples, each contains a list of field data(Tensor):
            tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx
    Returns:
        output: list of field batched data, which shape is [batch, max_length]
    """

    # 在这里主要做 padding 的工作
    batch_size = len(batch)
    max_length = max(x[0].shape[0] for x in batch)
    output = []

    for field_idx in range(6):
        # 这里处理包含 tokens, type_ids, start_labels, end_labels, start_label_mask, end_label_mask

        pad_output = torch.full([batch_size, max_length], 0, dtype=batch[0][field_idx].dtype)
        for sample_idx in range(batch_size):
            data = batch[sample_idx][field_idx]

            # 赋值为 0
            pad_output[sample_idx][: data.shape[0]] = data
        output.append(pad_output)

    # match_labels 进行 padding
    pad_match_labels = torch.zeros([batch_size, max_length, max_length], dtype=torch.long)

    for sample_idx in range(batch_size):
        data = batch[sample_idx][6]
        pad_match_labels[sample_idx, : data.shape[1], : data.shape[1]] = data
    output.append(pad_match_labels)

    # 另外的 sample_idx 和 label_idx 不需要 padding 所以单独处理
    output.append(torch.stack([x[-2] for x in batch]))
    output.append(torch.stack([x[-1] for x in batch]))

    return output
