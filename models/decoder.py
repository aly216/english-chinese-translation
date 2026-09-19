import torch
import torch.nn as nn
import torch.nn.functional as F
from config import device


# 没有注意力机制的解码器
class DecoderRNN(nn.Module):
    def __init__(self, output_size, hidden_size):
        """
        初始化解码器
        :param output_size: 解码器输出维度
        :param hidden_size: 隐藏层大小
        """
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
