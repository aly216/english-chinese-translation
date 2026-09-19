import torch
from torch.utils.data import Dataset, DataLoader
from config import device, EOS_token
from data_utils.preprocess import english_word2index, chinese_char2index, my_pairs


# 构建数据集对象
class MyPairsDataset(Dataset):
    def __init__(self, my_pairs):
        self.my_pairs = my_pairs
        self.sample_len = len(my_pairs)

    def __len__(self):
        return self.sample_len

    def __getitem__(self, idx):
        idx = min(max(idx, 0), self.sample_len - 1)
        x = self.my_pairs[idx][0]  # 英文
        y = self.my_pairs[idx][1]  # 中文

        x = [english_word2index[word] for word in x.split(' ')]
        x.append(EOS_token)
        tensor_x = torch.tensor(x, dtype=torch.long, device=device)
        y = [chinese_char2index[char] for char in y]  # 中文按字切分
        y.append(EOS_token)
        tensor_y = torch.tensor(y, dtype=torch.long, device=device)
        return tensor_x, tensor_y


# 构建数据加载器对象
def get_dataloader():
    my_dataset = MyPairsDataset(my_pairs)
    my_dataloader = DataLoader(
        my_dataset,
        batch_size=1,
        shuffle=True
    )
    return my_dataloader
