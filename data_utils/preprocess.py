import re
from config import data_path, SOS_token, EOS_token, MAX_LENGTH


# 英文清洗：转小写、标点前后加空格、只保留字母和 .!?
def normalize_english(s):
    s = s.lower().strip()
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# 中文清洗：去掉所有空白，每个汉字（含标点）当作一个 token
def normalize_chinese(s):
    s = s.strip()
    s = re.sub(r"\s+", "", s)
    return s


def my_get_data():
    with open(data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

        # 跳过第1行表头（english\tchinese），并丢弃不含制表符的异常行
        lines = lines[1:]
        my_pairs = []
        for line in lines:
            if "\t" not in line:
                continue
            parts = line.split("\t")
            my_pairs.append([normalize_english(parts[0]), normalize_chinese(parts[1])])

        # 过滤掉过长句子：英文按单词数，中文按字数
        my_pairs = [pair for pair in my_pairs
                    if len(pair[0].split(' ')) < MAX_LENGTH and len(pair[1]) < MAX_LENGTH]

        english_word2index = {'SOS': SOS_token, 'EOS': EOS_token}
        english_vocab_size = 2
        chinese_char2index = {'SOS': SOS_token, 'EOS': EOS_token}
        chinese_vocab_size = 2

        for pair in my_pairs:
            for word in pair[0].split(' '):
                if word not in english_word2index:
                    english_word2index[word] = english_vocab_size
                    english_vocab_size += 1
            for char in pair[1]:  # 中文按字切分，每个字一个 token
                if char not in chinese_char2index:
                    chinese_char2index[char] = chinese_vocab_size
                    chinese_vocab_size += 1

        english_index2word = {i: word for word, i in english_word2index.items()}
        chinese_index2char = {i: char for char, i in chinese_char2index.items()}

        return (english_word2index, english_index2word, english_vocab_size,
                chinese_char2index, chinese_index2char, chinese_vocab_size, my_pairs)


# 模块级执行一次，生成全局词表，供 dataset / training / evaluation 引用
english_word2index, english_index2word, english_vocab_size, chinese_char2index, chinese_index2char, chinese_vocab_size, my_pairs = my_get_data()
