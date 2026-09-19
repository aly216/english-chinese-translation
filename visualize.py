import torch
import matplotlib.pyplot as plt

# 中文字体（避免坐标轴中文显示为方块）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

from config import device, EOS_token, PATH1, PATH2
from data_utils.preprocess import english_word2index, english_vocab_size, chinese_vocab_size
from models.encoder import EncoderRNN
from models.attention_decoder import AttentionDecoderRNN
from evaluation.predict import evaluate_seq2seq


def show_attention(sentence, encoder, decoder, save_path=None):
    """
    可视化注意力权重：画出「输入英文单词 ↔ 生成中文字符」的注意力热力图。
    颜色越亮，表示生成该中文字符时越"关注"对应的英文词。
    """
    words = sentence.lower().split()
    tmpx = [english_word2index[w] for w in words]
    tmpx.append(EOS_token)
    tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

    decoder_chars, attentions = evaluate_seq2seq(tensor_x, encoder, decoder)

    # attentions 形状 [生成步数, MAX_LENGTH]，只取真实输入长度，并与输出字对齐
    attn = attentions[:len(decoder_chars), :len(words)].cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(6, len(words)), max(4, len(decoder_chars) * 0.5)))
    im = ax.imshow(attn, cmap='Blues', aspect='auto')

    ax.set_xticks(range(len(words)))
    ax.set_xticklabels(words, rotation=45)
    ax.set_yticks(range(len(decoder_chars)))
    ax.set_yticklabels(decoder_chars)

    ax.set_xlabel('输入英文单词')
    ax.set_ylabel('生成中文字符')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    encoder = EncoderRNN(english_vocab_size, 256).to(device)
    decoder = AttentionDecoderRNN(chinese_vocab_size, 256).to(device)
    encoder.load_state_dict(torch.load(PATH1, map_location=device, weights_only=True), strict=False)
    decoder.load_state_dict(torch.load(PATH2, map_location=device, weights_only=True), strict=False)

    for i, s in enumerate(['i am a student', 'i love you'], start=1):
        show_attention(s, encoder, decoder, save_path=f'./img/attention_{i}.png')


if __name__ == '__main__':
    main()
