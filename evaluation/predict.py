import torch
from config import device, SOS_token, EOS_token, MAX_LENGTH, PATH1, PATH2
from data_utils.dataset import get_dataloader
from data_utils.preprocess import english_word2index, chinese_index2char, english_vocab_size, chinese_vocab_size
from models.encoder import EncoderRNN
from models.attention_decoder import AttentionDecoderRNN


def evaluate_seq2seq(x, my_encoder, my_attn_decoder):
    with torch.no_grad():
        encoder_hidden = my_encoder.init_hidden()
        encoder_output, encoder_hidden = my_encoder(x, encoder_hidden)
        encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder.hidden_size, device=device)
        for i in range(x.shape[1]):
            encoder_output_c[i] = encoder_output[0, i]

        decoder_hidden = encoder_hidden
        input_y = torch.tensor([[SOS_token]], device=device)
        decoder_chars = []
        # 注意力权重矩阵要建在同一个设备上
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH, device=device)
        for i in range(MAX_LENGTH):
            # 解码器前向传播
            output_y, decoder_hidden, attn_weights = my_attn_decoder(input_y, decoder_hidden, encoder_output_c)
            decoder_attentions[i] = attn_weights
            topv, topi = output_y.topk(1)
            if topi.squeeze().item() == EOS_token:
                break
            else:
                decoder_chars.append(chinese_index2char[topi.squeeze().item()])
            input_y = topi.detach()

    return decoder_chars, decoder_attentions[:i + 1]


# 模型调用
def dm_test_seq2seq():
    my_dataloader = get_dataloader()
    my_encoder_rnn = EncoderRNN(english_vocab_size, 256).to(device)
    my_attn_decoder = AttentionDecoderRNN(chinese_vocab_size, 256).to(device)
    my_encoder_rnn.load_state_dict(torch.load(PATH1, map_location=device, weights_only=True), strict=False)
    my_attn_decoder.load_state_dict(torch.load(PATH2, map_location=device, weights_only=True), strict=False)
    my_sample_pairs = [
        ['i am a student', '我是学生'],
        ['i love you', '我爱你'],
        ['thank you very much', '非常感谢你'],
    ]
    for index, pair in enumerate(my_sample_pairs):
        x = pair[0]
        y = pair[1]
        tmpx = [english_word2index[w] for w in x.split()]
        tmpx.append(EOS_token)
        tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)
        decoder_chars, decoder_attentions = evaluate_seq2seq(tensor_x, my_encoder_rnn, my_attn_decoder)
        output_sentence = ''.join(decoder_chars)  # 中文无空格，直接拼接
        print(f'输入英文：{x}，参考答案：{y}')
        print(f'模型输出：{output_sentence}')
