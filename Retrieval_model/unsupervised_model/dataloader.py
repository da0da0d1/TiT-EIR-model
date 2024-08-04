from torch.utils.data import DataLoader, Dataset
import torch


class EViTPair(Dataset):

    def __init__(self, img_data, img_data_b, huffman_feature, huffman_feature_b, transform=None):
        self.img_data = img_data
        self.img_data_b = img_data_b
        self.huffman_feature = huffman_feature
        self.huffman_feature_b = huffman_feature_b
        self.transform = transform

    def __getitem__(self, index):
        img = self.img_data[index]
        img_b = self.img_data_b[index]
        huffman = torch.tensor(self.huffman_feature[index])
        huffman_b = torch.tensor(self.huffman_feature_b[index])

        if self.transform is not None:
            im_1 = self.transform(img)
            im_1_b = self.transform(img_b)

        return im_1, im_1_b, huffman, huffman_b

    def __len__(self):
        return len(self.img_data)
