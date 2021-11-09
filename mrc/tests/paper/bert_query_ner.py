# encoding: utf-8


import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel

from easytext.utils.seed_util import set_seed
from mrc.tests.paper.classifier import MultiNonLinearClassifier


class BertQueryNER(BertPreTrainedModel):
    def __init__(self, config):
        super(BertQueryNER, self).__init__(config)

        # bert 模型
        self.bert = BertModel(config)

        # self.start_outputs = nn.Linear(config.hidden_size, 2)
        # self.end_outputs = nn.Linear(config.hidden_size, 2)

        # start
        self.start_outputs = nn.Linear(config.hidden_size, 1)
        # end
        self.end_outputs = nn.Linear(config.hidden_size, 1)

        # span
        self.span_embedding = MultiNonLinearClassifier(config.hidden_size * 2, 1, config.mrc_dropout)
        # self.span_embedding = SingleLinearClassifier(config.hidden_size * 2, 1)

        self.hidden_size = config.hidden_size

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
            input_ids: bert input tokens, tensor of shape [seq_len]
            token_type_ids: 0 for query, 1 for context, tensor of shape [seq_len]
            attention_mask: attention mask, tensor of shape [seq_len]
        Returns:
            start_logits: start/non-start probs of shape [seq_len]
            end_logits: end/non-end probs of shape [seq_len]
            match_logits: start-end-match probs of shape [seq_len, 1]
        """

        # 执行 bert
        bert_outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # 获取 sequence 输出
        sequence_heatmap = bert_outputs[0]  # [batch, seq_len, hidden]
        batch_size, seq_len, hid_size = sequence_heatmap.size()

        # 得到 start logits
        start_logits = self.start_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len]

        # 得到 end logits
        end_logits = self.end_outputs(sequence_heatmap).squeeze(-1)  # [batch, seq_len]

        # 将每一个 i 与 j 连接在一起， 所以是 N*N的拼接，使用了 expand, 进行 两个方向的扩展
        # start 和 end 的 logits 有必要存在吗？??? 还是说，增加了一些约束？
        # for every position $i$ in sequence, should concate $j$ to
        # predict if $i$ and $j$ are start_pos and end_pos for an entity.
        # [batch, seq_len, seq_len, hidden]
        start_extend = sequence_heatmap.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # [batch, seq_len, seq_len, hidden]
        end_extend = sequence_heatmap.unsqueeze(1).expand(-1, seq_len, -1, -1)
        # [batch, seq_len, seq_len, hidden*2]
        span_matrix = torch.cat([start_extend, end_extend], 3)
        # [batch, seq_len, seq_len]
        span_logits = self.span_embedding(span_matrix).squeeze(-1)

        return start_logits, end_logits, span_logits
