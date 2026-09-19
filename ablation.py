import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

# 中文字体（避免坐标轴中文显示为方块）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

from config import device, SOS_token, EOS_token, MAX_LENGTH, my_lr, epochs, teacher_forcing_ratio
from data_utils.dataset import get_dataloader
from data_utils.preprocess import english_vocab_size, chinese_vocab_size, english_word2index, chinese_index2char
from models.encoder import EncoderRNN
from models.attention_decoder import AttentionDecoderRNN
from models.decoder import DecoderRNN


def encode(encoder, x):
    """编码器前向，返回填充到 MAX_LENGTH 的编码输出 + 最终隐藏状态"""
    encoder_hidden = encoder.init_hidden()
    encoder_output, encoder_hidden = encoder(x, encoder_hidden)
    encoder_output_c = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    for i in range(x.shape[1]):
        encoder_output_c[i] = encoder_output[0, i]
    return encoder_output_c, encoder_hidden


def train_one_step(x, y, encoder, decoder, opt_enc, opt_dec, loss_fn, use_attention):
    """单步训练，use_attention 决定调用哪种解码器接口"""
    encoder_output_c, encoder_hidden = encode(encoder, x)
    decoder_hidden = encoder_hidden
    input_y = torch.tensor([[SOS_token]], device=device)
    my_loss = 0.0
    y_len = y.shape[1]
    use_tf = random.random() < teacher_forcing_ratio
    for i in range(y_len):
        if use_attention:
            output, decoder_hidden, _ = decoder(input_y, decoder_hidden, encoder_output_c)
        else:
            output, decoder_hidden = decoder(input_y, decoder_hidden)
        target_y = y[0][i].view(1)
        my_loss += loss_fn(output, target_y)
        if use_tf:
            input_y = y[0][i].view(1, -1)
        else:
            topv, topi = output.topk(1)
            if topi.squeeze().item() == EOS_token:
                break
            input_y = topi.detach()

    opt_enc.zero_grad()
    opt_dec.zero_grad()
    my_loss.backward()
    opt_enc.step()
    opt_dec.step()
    return my_loss.item() / y_len


def greedy_decode(x, encoder, decoder, use_attention):
    with torch.no_grad():
        encoder_output_c, encoder_hidden = encode(encoder, x)
        decoder_hidden = encoder_hidden
        input_y = torch.tensor([[SOS_token]], device=device)
        chars = []
        for _ in range(MAX_LENGTH):
            if use_attention:
                output, decoder_hidden, _ = decoder(input_y, decoder_hidden, encoder_output_c)
            else:
                output, decoder_hidden = decoder(input_y, decoder_hidden)
            topv, topi = output.topk(1)
            if topi.squeeze().item() == EOS_token:
                break
            chars.append(chinese_index2char[topi.squeeze().item()])
            input_y = topi.detach()
    return ''.join(chars)  # 中文无空格，直接拼接


def _checkpoint_paths(name):
    """返回该变体的权重 / 损失曲线保存路径"""
    return (f'./checkpoints/ablation_{name}_encoder.pth',
            f'./checkpoints/ablation_{name}_decoder.pth',
            f'./checkpoints/ablation_{name}_losses.json')


def train_or_load(decoder_cls, use_attention, name, tag, force_retrain=False):
    """有已训练权重就加载，没有就训练并保存。返回 (每 epoch 损失列表, encoder, decoder)"""
    enc_path, dec_path, loss_path = _checkpoint_paths(name)

    # 已训练过 → 直接加载，跳过训练
    if not force_retrain and os.path.exists(enc_path) and os.path.exists(dec_path) and os.path.exists(loss_path):
        print(f'{tag}：检测到已训练权重，直接加载（想重训请删 checkpoints/ablation_{name}_* 或设 force_retrain=True）')
        encoder = EncoderRNN(english_vocab_size, 256).to(device)
        decoder = decoder_cls(chinese_vocab_size, 256).to(device)
        encoder.load_state_dict(torch.load(enc_path, map_location=device, weights_only=True))
        decoder.load_state_dict(torch.load(dec_path, map_location=device, weights_only=True))
        with open(loss_path, 'r', encoding='utf-8') as f:
            epoch_losses = json.load(f)
        return epoch_losses, encoder, decoder

    # 未训练 → 训练并保存
    dataloader = get_dataloader()
    encoder = EncoderRNN(english_vocab_size, 256).to(device)
    decoder = decoder_cls(chinese_vocab_size, 256).to(device)
    opt_enc = optim.Adam(encoder.parameters(), lr=my_lr)
    opt_dec = optim.Adam(decoder.parameters(), lr=my_lr)
    loss_fn = nn.NLLLoss()

    epoch_losses = []
    for epoch in range(1, epochs + 1):
        total = 0.0
        for item, (x, y) in enumerate(tqdm(dataloader, desc=f'{tag} epoch {epoch}'), start=1):
            total += train_one_step(x, y, encoder, decoder, opt_enc, opt_dec, loss_fn, use_attention)
        avg = total / item
        epoch_losses.append(avg)
        print(f'{tag} epoch {epoch}: 平均损失 = {avg:.4f}')

    torch.save(encoder.state_dict(), enc_path)
    torch.save(decoder.state_dict(), dec_path)
    with open(loss_path, 'w', encoding='utf-8') as f:
        json.dump(epoch_losses, f)
    print(f'{tag}：已保存到 checkpoints/ablation_{name}_*')
    return epoch_losses, encoder, decoder


def main():
    print('=== 「有注意力」模型（训练或加载）===')
    attn_losses, attn_enc, attn_dec = train_or_load(AttentionDecoderRNN, use_attention=True, name='attn', tag='有注意力')
    print('=== 「无注意力」模型（训练或加载）===')
    plain_losses, plain_enc, plain_dec = train_or_load(DecoderRNN, use_attention=False, name='plain', tag='无注意力')

    # 损失曲线对比
    plt.figure()
    plt.plot(range(1, epochs + 1), attn_losses, marker='o', label='有注意力')
    plt.plot(range(1, epochs + 1), plain_losses, marker='s', label='无注意力')
    plt.xlabel('epoch')
    plt.ylabel('平均损失')
    plt.title('有注意力 vs 无注意力 训练损失对比')
    plt.legend()
    plt.savefig('./img/ablation_loss.png')
    plt.show()

    # 翻译样例对比
    samples = ['i am a student', 'i love you']
    print('\n=== 翻译对比 ===')
    for s in samples:
        tmpx = [english_word2index[w] for w in s.lower().split()]
        tmpx.append(EOS_token)
        tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)
        out_attn = greedy_decode(tensor_x, attn_enc, attn_dec, use_attention=True)
        out_plain = greedy_decode(tensor_x, plain_enc, plain_dec, use_attention=False)
        print(f'输入: {s}')
        print(f'  有注意力: {out_attn}')
        print(f'  无注意力: {out_plain}')


if __name__ == '__main__':
    main()
