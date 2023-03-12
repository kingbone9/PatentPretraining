#!/usr/bin/python
# author kingbone
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import json
import logging
import os
from sklearn.model_selection import train_test_split
import collections
import six

ipc2id = {
    'A01': 0,
    'A11': 1,
    'A21': 2,
    'A22': 3,
    'A23': 4,
    'A24': 5,
    'A41': 6,
    'A42': 7,
    'A43': 8,
    'A44': 9,
    'A45': 10,
    'A46': 11,
    'A47': 12,
    'A61': 13,
    'A62': 14,
    'A63': 15,
    'A99': 16,
    'B01': 17,
    'B02': 18,
    'B03': 19,
    'B04': 20,
    'B05': 21,
    'B06': 22,
    'B07': 23,
    'B08': 24,
    'B09': 25,
    'B21': 26,
    'B22': 27,
    'B23': 28,
    'B24': 29,
    'B25': 30,
    'B26': 31,
    'B27': 32,
    'B28': 33,
    'B29': 34,
    'B30': 35,
    'B31': 36,
    'B32': 37,
    'B33': 38,
    'B41': 39,
    'B42': 40,
    'B43': 41,
    'B44': 42,
    'B60': 43,
    'B61': 44,
    'B62': 45,
    'B63': 46,
    'B64': 47,
    'B65': 48,
    'B66': 49,
    'B67': 50,
    'B68': 51,
    'B81': 52,
    'B82': 53,
    'C01': 54,
    'C02': 55,
    'C03': 56,
    'C04': 57,
    'C05': 58,
    'C06': 59,
    'C07': 60,
    'C08': 61,
    'C09': 62,
    'C10': 63,
    'C11': 64,
    'C12': 65,
    'C13': 66,
    'C14': 67,
    'C21': 68,
    'C22': 69,
    'C23': 70,
    'C25': 71,
    'C30': 72,
    'C40': 73,
    'D01': 74,
    'D02': 75,
    'D03': 76,
    'D04': 77,
    'D05': 78,
    'D06': 79,
    'D07': 80,
    'D10': 81,
    'D21': 82,
    'E01': 83,
    'E02': 84,
    'E03': 85,
    'E04': 86,
    'E05': 87,
    'E06': 88,
    'E21': 89,
    'F01': 90,
    'F02': 91,
    'F03': 92,
    'F04': 93,
    'F05': 94,
    'F15': 95,
    'F16': 96,
    'F17': 97,
    'F21': 98,
    'F22': 99,
    'F23': 100,
    'F24': 101,
    'F25': 102,
    'F26': 103,
    'F27': 104,
    'F28': 105,
    'F41': 106,
    'F42': 107,
    'G01': 108,
    'G02': 109,
    'G03': 110,
    'G04': 111,
    'G05': 112,
    'G06': 113,
    'G07': 114,
    'G08': 115,
    'G09': 116,
    'G10': 117,
    'G11': 118,
    'G12': 119,
    'G16': 120,
    'G21': 121,
    'H01': 122,
    'H02': 123,
    'H03': 124,
    'H04': 125,
    'H05': 126,
    'Y02': 127,
    'Y04': 128,
    'Y10': 129
}

id2ipc = {_id: _label for _label, _id in list(ipc2id.items())}


class Vocab:
    """
    根据本地的vocab文件，构造一个词表
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    print(len(vocab))  # 返回词表长度
    """
    UNK = '[UNK]'

    def __init__(self, vocab_path):
        self.stoi = {}
        self.itos = []
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for i, word in enumerate(f):
                w = word.strip('\n')
                self.stoi[w] = i
                self.itos.append(w)

    def __getitem__(self, token):
        return self.stoi.get(token, self.stoi.get(Vocab.UNK))

    def __len__(self):
        return len(self.itos)


def build_vocab(vocab_path):
    """
    vocab = Vocab()
    print(vocab.itos)  # 得到一个列表，返回词表中的每一个词；
    print(vocab.itos[2])  # 通过索引返回得到词表中对应的词；
    print(vocab.stoi)  # 得到一个字典，返回词表中每个词的索引；
    print(vocab.stoi['我'])  # 通过单词返回得到词表中对应的索引
    """
    return Vocab(vocab_path)


def pad_sequence(sequences, batch_first=False, max_len=None, padding_value=0):
    """
    对一个List中的元素进行padding
    Pad a list of variable length Tensors with ``padding_value``
    a = torch.ones(25)
    b = torch.ones(22)
    c = torch.ones(15)
    pad_sequence([a, b, c],max_len=None).size()
    torch.Size([25, 3])
        sequences:
        batch_first: 是否把batch_size放到第一个维度
        padding_value:
        max_len :
                当max_len = 50时，表示以某个固定长度对样本进行padding，多余的截掉；
                当max_len=None是，表示以当前batch中最长样本的长度对其它进行padding；
    Returns:
    """
    if max_len is None:
        max_len = max([s.size(0) for s in sequences])
    out_tensors = []
    for tensor in sequences:
        if tensor.size(0) < max_len:
            tensor = torch.cat([tensor, torch.tensor([padding_value] * (max_len - tensor.size(0)))], dim=0)
        else:
            tensor = tensor[:max_len]
        out_tensors.append(tensor)
    out_tensors = torch.stack(out_tensors, dim=1)
    if batch_first:
        return out_tensors.transpose(0, 1)
    return out_tensors
