import torch
import torch.nn as nn
import torch.nn.functional as F
from config import device, MAX_LENGTH


class AttentionDecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size, dropout=0.1, max_len=MAX_LENGTH):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout_p = dropout
        self.max_len = max_len
        # 参1：拼接编码器输出和解码器隐藏状态，参2：注意力的权重分布
        self.attn = nn.Linear(self.hidden_size * 2, self.max_len)
        # 将词嵌入层的输出和注意力机制的输出融合
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, encoder_outputs):
        """
        前向传播
        :param input: 输入序列，{batch_size, 1}
        :param hidden: 解码器隐藏状态，{1, batch_size, hidden_size}
        :param encoder_outputs: 编码器输出，{batch_size, max_len, hidden_size}
        """
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        attn_weights = self.attn(torch.cat((embedded[0], hidden[0]), dim=1))
        attn_weights = F.softmax(attn_weights, dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
        output = torch.cat((embedded[0], attn_applied.squeeze(0)), dim=1)
        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden, attn_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
