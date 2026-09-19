import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import (device, SOS_token, EOS_token, MAX_LENGTH,
                    my_lr, epochs, teacher_forcing_ratio,
                    print_interval_num, plot_interval_num)
from data_utils.dataset import get_dataloader
from data_utils.preprocess import english_vocab_size, chinese_vocab_size
from models.encoder import EncoderRNN
from models.attention_decoder import AttentionDecoderRNN


def train_iters(x, y, my_encoder, my_attn_decoder, myadam_encoder, myadam_decoder, my_crossentropy_loss):
    """
    训练模型
    :param x: 输入英文序列，{batch_size, seq_len}
    :param y: 输出中文序列，{batch_size, seq_len}
    :param my_encoder: 编码器模型
    :param my_attn_decoder: 注意力机制的解码器模型
    :param myadam_encoder: 编码器优化器
    :param myadam_decoder: 解码器优化器
    :param my_crossentropy_loss: 交叉熵损失函数
    """
    encoder_hidden = my_encoder.init_hidden()
    encoder_output, encoder_hidden = my_encoder(x, encoder_hidden)
    encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder.hidden_size, device=device)
    for i in range(x.shape[1]):
        encoder_output_c[i] = encoder_output[0, i]
    # 初始化解码器隐藏状态
    # 初始化输入序列（形状必须是{batch_size, 1}）
    decoder_hidden = encoder_hidden
    input_y = torch.tensor([[SOS_token]], device=device)

    my_loss = 0.0
    y_len = y.shape[1]
    # 决定是否使用教师强制训练
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        for i in range(y_len):
            output_y, decoder_hidden, attn_weights = my_attn_decoder(input_y, decoder_hidden, encoder_output_c)
            target_y = y[0][i].view(1)
            my_loss += my_crossentropy_loss(output_y, target_y)
            input_y = y[0][i].view(1, -1)
    else:
        for i in range(y_len):
            output_y, decoder_hidden, attn_weights = my_attn_decoder(input_y, decoder_hidden, encoder_output_c)
            target_y = y[0][i].view(1)
            my_loss += my_crossentropy_loss(output_y, target_y)
            topv, topi = output_y.topk(1)
            if topi.squeeze().item() == EOS_token:
                break
            input_y = topi.detach()

    myadam_encoder.zero_grad()
    myadam_decoder.zero_grad()
    my_loss.backward()
    myadam_encoder.step()
    myadam_decoder.step()

    return my_loss.item() / y_len


def train_seq2seq():
    my_dataloader = get_dataloader()
    my_encoder_rnn = EncoderRNN(english_vocab_size, 256).to(device)
    my_attn_decoder = AttentionDecoderRNN(chinese_vocab_size, 256).to(device)
    myadam_encoder = optim.Adam(my_encoder_rnn.parameters(), lr=my_lr)
    myadam_decoder = optim.Adam(my_attn_decoder.parameters(), lr=my_lr)
    my_crossentropy_loss = nn.NLLLoss()

    plot_loss_list = []
    for epoch_idx in range(1, epochs + 1):
        print_loss_total = 0.0
        plot_loss_total = 0.0
        start_time = time.time()
        for item, (x, y) in enumerate(tqdm(my_dataloader), start=1):
            my_loss = train_iters(x, y, my_encoder_rnn, my_attn_decoder, myadam_encoder, myadam_decoder, my_crossentropy_loss)
            print_loss_total += my_loss
            plot_loss_total += my_loss

            if item % print_interval_num == 0:
                print_loss_avg = print_loss_total / print_interval_num
                print_loss_total = 0.0
                # 打印训练信息，批次，损失，时间
                print(f'批次{item}，损失{print_loss_avg:.4f}，时间{time.time() - start_time:.2f}')

            if item % plot_interval_num == 0:
                plot_loss_avg = plot_loss_total / plot_interval_num
                plot_loss_list.append(plot_loss_avg)
                plot_loss_total = 0.0

        torch.save(my_encoder_rnn.state_dict(), f'./checkpoints/my_encoder_rnn_{epoch_idx}.pth')
        torch.save(my_attn_decoder.state_dict(), f'./checkpoints/my_attn_decoder_{epoch_idx}.pth')

        plt.figure()
        plt.plot(plot_loss_list)
        plt.savefig(f'./img/plot_loss_{epoch_idx}.png')
        plt.show()

    return plot_loss_list
