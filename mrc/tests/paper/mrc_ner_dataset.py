# encoding: utf-8


import json
import torch
from tokenizers import BertWordPieceTokenizer
from torch.utils.data import Dataset


class MRCNERDataset(Dataset):
    """
    原始的 paper 中的
    MRC NER Dataset
    Args:
        json_path: path to mrc-ner style json
        tokenizer: BertTokenizer
        max_length: int, max length of query+context
        possible_only: if True, only use possible samples that contain answer for the query/context
        is_chinese: is chinese dataset
    """
    def __init__(self, json_path, tokenizer: BertWordPieceTokenizer, max_length: int = 128, possible_only=False,
                 is_chinese=False, pad_to_maxlen=False):
        self.all_data = json.load(open(json_path, encoding="utf-8"))
        self.tokenzier = tokenizer
        self.max_length = max_length
        self.possible_only = possible_only
        if self.possible_only:
            self.all_data = [
                x for x in self.all_data if x["start_position"]
            ]
        self.is_chinese = is_chinese
        self.pad_to_maxlen = pad_to_maxlen

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, item):
        """
        Args:
            item: int, idx
        Returns:
            tokens: tokens of query + context, [seq_len]
            token_type_ids: token type ids, 0 for query, 1 for context, [seq_len]
            start_labels: start labels of NER in tokens, [seq_len]
            end_labels: end labelsof NER in tokens, [seq_len]
            label_mask: label mask, 1 for counting into loss, 0 for ignoring. [seq_len]
            match_labels: match labels, [seq_len, seq_len]
            sample_idx: sample id
            label_idx: label id

        """
        data = self.all_data[item]
        tokenizer = self.tokenzier

        # sample_idx 没什么用
        qas_id = data.get("qas_id", "0.0")
        sample_idx, label_idx = qas_id.split(".")
        sample_idx = torch.LongTensor([int(sample_idx)])
        label_idx = torch.LongTensor([int(label_idx)])

        query = data["query"]
        context = data["context"]
        start_positions = data["start_position"]
        end_positions = data["end_position"]

        if self.is_chinese:
            context = "".join(context.split())
            end_positions = [x+1 for x in end_positions]
        else:
            # add space offsets
            words = context.split()
            start_positions = [x + sum([len(w) for w in words[:x]]) for x in start_positions]
            end_positions = [x + sum([len(w) for w in words[:x + 1]]) for x in end_positions]

        # encode
        query_context_tokens = tokenizer.encode(query, context, add_special_tokens=True)
        tokens = query_context_tokens.ids
        type_ids = query_context_tokens.type_ids

        # 每一个 token id 对应一个 offset, 注意 CLS, SEP 对应的 offset 都是 (0, 0)
        # 这也就是说对于 SEP 拼接的，每一个部分的 offset 是 0 重新开始计数的，0 就是 SEP
        offsets = query_context_tokens.offsets

        # find new start_positions/end_positions, considering
        # 1. we add query tokens at the beginning
        # 2. word-piece tokenize

        # 因为 query 和 context 拼接在一起了，所以 start_position 和 end_position 的位置要重新映射
        origin_offset2token_idx_start = {}
        origin_offset2token_idx_end = {}

        for token_idx in range(len(tokens)):
            # query 的需要过滤
            if type_ids[token_idx] == 0:
                continue

            # 获取每一个 token_start 和 end
            token_start, token_end = offsets[token_idx]

            # skip [CLS] or [SEP], offset 中 (0, 0) 表示的就是 CLS 或者 SEP
            if token_start == token_end == 0:
                continue

            # token_start 对应的就是 context 中的实际位置，与 start_position 与 end_position 是对应的
            # token_idx 是 query 和 context 拼接在一起后的 index，所以 这就是 start_position 映射后的位置
            origin_offset2token_idx_start[token_start] = token_idx
            origin_offset2token_idx_end[token_end] = token_idx

        # 将原始数据中的  start_positions 映射到 拼接 query context 之后的位置
        new_start_positions = [origin_offset2token_idx_start[start] for start in start_positions]
        new_end_positions = [origin_offset2token_idx_end[end] for end in end_positions]

        # 对于 mask, query, CLS, SEP 都被 mask 掉
        label_mask = [
            (0 if type_ids[token_idx] == 0 or offsets[token_idx] == (0, 0) else 1)
            for token_idx in range(len(tokens))
        ]
        start_label_mask = label_mask.copy()
        end_label_mask = label_mask.copy()

        # the start/end position must be whole word
        if not self.is_chinese:
            for token_idx in range(len(tokens)):
                current_word_idx = query_context_tokens.words[token_idx]
                next_word_idx = query_context_tokens.words[token_idx+1] if token_idx+1 < len(tokens) else None
                prev_word_idx = query_context_tokens.words[token_idx-1] if token_idx-1 > 0 else None
                if prev_word_idx is not None and current_word_idx == prev_word_idx:
                    start_label_mask[token_idx] = 0
                if next_word_idx is not None and current_word_idx == next_word_idx:
                    end_label_mask[token_idx] = 0

        assert all(start_label_mask[p] != 0 for p in new_start_positions)
        assert all(end_label_mask[p] != 0 for p in new_end_positions)

        assert len(new_start_positions) == len(new_end_positions) == len(start_positions)
        assert len(label_mask) == len(tokens)

        # start_position label 序列
        start_labels = [(1 if idx in new_start_positions else 0)
                        for idx in range(len(tokens))]

        # end_position label 序列
        end_labels = [(1 if idx in new_end_positions else 0)
                      for idx in range(len(tokens))]

        # truncate
        tokens = tokens[: self.max_length]
        type_ids = type_ids[: self.max_length]
        start_labels = start_labels[: self.max_length]
        end_labels = end_labels[: self.max_length]
        start_label_mask = start_label_mask[: self.max_length]
        end_label_mask = end_label_mask[: self.max_length]

        # make sure last token is [SEP]
        # 保证最后一个 token 是 [SEP]
        sep_token = tokenizer.token_to_id("[SEP]")
        if tokens[-1] != sep_token:
            assert len(tokens) == self.max_length

            # 最后一个设置为 SEP
            tokens = tokens[: -1] + [sep_token]

            # 所有的 label 都要进行修改
            start_labels[-1] = 0
            end_labels[-1] = 0
            start_label_mask[-1] = 0
            end_label_mask[-1] = 0

        if self.pad_to_maxlen:
            tokens = self.pad(tokens, 0)
            type_ids = self.pad(type_ids, 1)
            start_labels = self.pad(start_labels)
            end_labels = self.pad(end_labels)
            start_label_mask = self.pad(start_label_mask)
            end_label_mask = self.pad(end_label_mask)

        seq_len = len(tokens)
        # 这里定义 (start_position, end_position) 的 pair 标签, 显然是二维的
        match_labels = torch.zeros([seq_len, seq_len], dtype=torch.long)

        for start, end in zip(new_start_positions, new_end_positions):
            if start >= seq_len or end >= seq_len:
                continue
            match_labels[start, end] = 1

        return [
            torch.LongTensor(tokens),
            torch.LongTensor(type_ids),
            torch.LongTensor(start_labels),
            torch.LongTensor(end_labels),
            torch.LongTensor(start_label_mask),
            torch.LongTensor(end_label_mask),
            match_labels,
            sample_idx,
            label_idx
        ]

    def pad(self, lst, value=0, max_length=None):
        max_length = max_length or self.max_length
        while len(lst) < max_length:
            lst.append(value)
        return lst


def run_dataset():
    """test dataset"""
    import os
    from datasets.collate_functions import collate_to_max_length
    from torch.utils.data import DataLoader
    # zh datasets
    # bert_path = "/mnt/mrc/chinese_L-12_H-768_A-12"
    # json_path = "/mnt/mrc/zh_msra/mrc-ner.test"
    # # json_path = "/mnt/mrc/zh_onto4/mrc-ner.train"
    # is_chinese = True

    # en datasets
    bert_path = "/mnt/mrc/bert-base-uncased"
    json_path = "/mnt/mrc/ace2004/mrc-ner.train"
    # json_path = "/mnt/mrc/genia/mrc-ner.train"
    is_chinese = False

    vocab_file = os.path.join(bert_path, "vocab.txt")
    tokenizer = BertWordPieceTokenizer(vocab_file=vocab_file)
    dataset = MRCNERDataset(json_path=json_path, tokenizer=tokenizer,
                            is_chinese=is_chinese)

    dataloader = DataLoader(dataset, batch_size=32,
                            collate_fn=collate_to_max_length)

    for batch in dataloader:
        for tokens, token_type_ids, start_labels, end_labels, start_label_mask, end_label_mask, match_labels, sample_idx, label_idx in zip(*batch):
            tokens = tokens.tolist()
            start_positions, end_positions = torch.where(match_labels > 0)
            start_positions = start_positions.tolist()
            end_positions = end_positions.tolist()
            if not start_positions:
                continue
            print("="*20)
            print(f"len: {len(tokens)}", tokenizer.decode(tokens, skip_special_tokens=False))
            for start, end in zip(start_positions, end_positions):
                print(str(sample_idx.item()), str(label_idx.item()) + "\t" + tokenizer.decode(tokens[start: end+1]))


if __name__ == '__main__':
    run_dataset()
