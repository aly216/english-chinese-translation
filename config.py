import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义常量
SOS_token = 0  # 开始符
EOS_token = 1  # 结束符
MAX_LENGTH = 100  # 最大长度
data_path = "./data/cmn-eng_10000.txt"

# 模型权重路径（推理用）
PATH1 = './checkpoints/my_encoder_rnn_10.pth'
PATH2 = './checkpoints/my_attn_decoder_10.pth'

# 训练超参数
my_lr = 1e-4
epochs = 10
teacher_forcing_ratio = 0.5
print_interval_num = 1000
plot_interval_num = 100
