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
    'A21': 1,
    'A22': 2,
    'A23': 3,
    'A24': 4,
    'A41': 5,
    'A43': 6,
    'A44': 7,
    'A45': 8,
    'A46': 9,
    'A47': 10,
    'A61': 11,
    'A62': 12,
    'A63': 13,
    'B01': 14,
    'B02': 15,
    'B03': 16,
    'B04': 17,
    'B05': 18,
    'B06': 19,
    'B07': 20,
    'B08': 21,
    'B09': 22,
    'B21': 23,
    'B22': 24,
    'B23': 25,
    'B24': 26,
    'B25': 27,
    'B26': 28,
    'B27': 29,
    'B28': 30,
    'B29': 31,
    'B30': 32,
    'B31': 33,
    'B32': 34,
    'B33': 35,
    'B41': 36,
    'B42': 37,
    'B43': 38,
    'B44': 39,
    'B60': 40,
    'B61': 41,
    'B62': 42,
    'B63': 43,
    'B64': 44,
    'B65': 45,
    'B66': 46,
    'B67': 47,
    'B68': 48,
    'B81': 49,
    'B82': 50,
    'C01': 51,
    'C02': 52,
    'C03': 53,
    'C04': 54,
    'C05': 55,
    'C06': 56,
    'C07': 57,
    'C08': 58,
    'C09': 59,
    'C10': 60,
    'C11': 61,
    'C12': 62,
    'C14': 63,
    'C21': 64,
    'C22': 65,
    'C23': 66,
    'C25': 67,
    'C30': 68,
    'D01': 69,
    'D02': 70,
    'D03': 71,
    'D04': 72,
    'D05': 73,
    'D06': 74,
    'D07': 75,
    'D10': 76,
    'D21': 77,
    'E01': 78,
    'E02': 79,
    'E03': 80,
    'E04': 81,
    'E05': 82,
    'E06': 83,
    'E21': 84,
    'F01': 85,
    'F02': 86,
    'F03': 87,
    'F04': 88,
    'F05': 89,
    'F15': 90,
    'F16': 91,
    'F17': 92,
    'F21': 93,
    'F22': 94,
    'F23': 95,
    'F24': 96,
    'F25': 97,
    'F26': 98,
    'F27': 99,
    'F28': 100,
    'F41': 101,
    'F42': 102,
    'G01': 103,
    'G02': 104,
    'G03': 105,
    'G04': 106,
    'G05': 107,
    'G06': 108,
    'G07': 109,
    'G08': 110,
    'G09': 111,
    'G10': 112,
    'G11': 113,
    'G12': 114,
    'G16': 115,
    'G21': 116,
    'H01': 117,
    'H02': 118,
    'H03': 119,
    'H04': 120,
    'H05': 121,
    'Y02': 122,
    'Y04': 123,
    'Y10': 124
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
