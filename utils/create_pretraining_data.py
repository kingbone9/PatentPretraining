#!/usr/bin/python
# author kingbone
import logging
import random
from tqdm import tqdm
from .data_helper import build_vocab, pad_sequence, id2ipc, ipc2id
from .data_helper import pad_sequence
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd


def cache(func):
    """
    本修饰器的作用是将数据预处理后的结果进行缓存，下次使用时可直接载入！
    :param func:
    :return:
    """

    def wrapper(*args, **kwargs):
        filepath = kwargs['filepath']
        postfix = kwargs['postfix']
        data_path = filepath.split('.')[0] + '_' + postfix + '.pt'
        if not os.path.exists(data_path):
            logging.info(f"缓存文件 {data_path} 不存在，重新处理并缓存！")
            data = func(*args, **kwargs)
            with open(data_path, 'wb') as f:
                torch.save(data, f)
        else:
            logging.info(f"缓存文件 {data_path} 存在，直接载入缓存文件！")
            with open(data_path, 'rb') as f:
                data = torch.load(f)
        return data

    return wrapper


class LoadBertPretrainingDataset(object):
    r"""

    Arguments:

    """

    def __init__(self,
                 vocab_path='./vocab.txt',
                 tokenizer=None,
                 batch_size=32,
                 max_sen_len=None,
                 max_position_embeddings=512,
                 pad_index=0,
                 is_sample_shuffle=True,
                 random_state=2021,
                 data_name='wiki2',
                 masked_rate=0.15,
                 masked_token_rate=0.8,
                 masked_token_unchanged_rate=0.5,
                 whole_word_masked_rate=0.25):
        self.tokenizer = tokenizer
        self.vocab = build_vocab(vocab_path)
        self.PAD_IDX = pad_index
        self.SEP_IDX = self.vocab['[SEP]']
        self.CLS_IDX = self.vocab['[CLS]']
        self.MASK_IDS = self.vocab['[MASK]']
        self.batch_size = batch_size
        self.max_sen_len = max_sen_len
        self.max_position_embeddings = max_position_embeddings
        self.pad_index = pad_index
        self.is_sample_shuffle = is_sample_shuffle
        self.data_name = data_name
        self.masked_rate = masked_rate
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate
        self.random_state = random_state
        self.whole_word_masked_rate = whole_word_masked_rate
        self.label_num = len(ipc2id)
        random.seed(random_state)

    @staticmethod
    def get_next_sentence_sample(sentence, next_sentence, paragraphs):
        """
        本函数的作用是根据给定的连续两句话和对应的段落，返回NSP任务中的句子对和标签
        :param sentence:  str
        :param next_sentence: str
        :param paragraphs: [str,str,...,]
        :return: sentence A, sentence B, True
        """
        if random.random() < 0.5:  # 产生[0,1)之间的一个随机数
            is_next = True
        else:
            # 这里random.choice的作用是从一个list中随机选出一个元素
            # ①先从所有段落中随机出一个段落；
            # ②再从随机出的一个段落中随机出一句话；
            next_sentence = random.choice(random.choice(paragraphs))
            is_next = False
        return sentence, next_sentence, is_next

    @staticmethod
    def get_ipc_label(ipc_list, label_num):
        ipc_label = torch.zeros(label_num)
        for ipc in ipc_list:
            ipc_label[ipc2id[ipc]] = 1
        return ipc_label

    @staticmethod
    def get_word_span_list(whole_word_dict):
        word_span_list = list()
        for spans in whole_word_dict.values():
            for span in spans:
                word_span_list += [[idx for idx in range(span[0] + 1, span[-1] + 2)]]
        return word_span_list

    def replace_masked_tokens(self, token_ids, candidate_pred_positions, num_mlm_preds, word_span_list, word_idx):
        """
        本函数的作用是根据给定的token_ids、候选mask位置以及需要mask的数量来返回被mask后的token_ids以及标签信息
        :param token_ids:
        :param candidate_pred_positions:
        :param num_mlm_preds:
        :return:
        """
        pred_positions = []
        word_mask_tag = 0
        mlm_input_tokens_id = [token_id for token_id in token_ids]
        for mlm_pred_position in candidate_pred_positions:
            if word_mask_tag >= num_mlm_preds:
                break  # 如果已经mask的数量大于等于num_mlm_preds则停止mask
            if mlm_pred_position in pred_positions:
                continue
            masked_token_id = None
            # 80%的时间：将词替换为['MASK']词元，但这里是直接替换为['MASK']对应的id
            if random.random() < self.masked_token_rate:  # 0.8
                masked_token_id = self.MASK_IDS
            else:
                # 10%的时间：保持词不变
                if random.random() < self.masked_token_unchanged_rate:  # 0.5
                    masked_token_id = token_ids[mlm_pred_position]
                # 10%的时间：用随机词替换该词
                else:
                    masked_token_id = random.randint(0, len(self.vocab.stoi) - 1)
            if masked_token_id == self.MASK_IDS and mlm_pred_position in word_idx and random.random() < self.whole_word_masked_rate:
                for word_idx_list in word_span_list:
                    if mlm_pred_position in word_idx_list:
                        for w_idx in word_idx_list:
                            mlm_input_tokens_id[w_idx] = masked_token_id
                            pred_positions.append(w_idx)
                        break
            else:
                mlm_input_tokens_id[mlm_pred_position] = masked_token_id
                pred_positions.append(mlm_pred_position)  # 保留被mask位置的索引信息
            word_mask_tag += 1
        # 构造mlm任务中需要预测位置对应的正确标签，如果其没出现在pred_positions则表示该位置不是mask位置
        # 则在进行损失计算时需要忽略掉这些位置（即为PAD_IDX）；而如果其出现在mask的位置，则其标签为原始token_ids对应的id
        mlm_label = [self.PAD_IDX if idx not in pred_positions
                     else token_ids[idx] for idx in range(len(token_ids))]
        return mlm_input_tokens_id, mlm_label

    def get_masked_sample(self, token_ids, word_span_list):
        """
        本函数的作用是将传入的 一段token_ids的其中部分进行mask处理
        :param token_ids:         e.g. [101, 1031, 4895, 2243, 1033, 10029, 2000, 2624, 1031,....]
        :return: mlm_input_tokens_id:  [101, 1031, 103, 2243, 1033, 10029, 2000, 103,  1031, ...]
                           mlm_label:  [ 0,   0,   4895,  0,    0,    0,    0,   2624,  0,...]
        """
        word_idx = list()
        for l in word_span_list:
            word_idx += l
        word_idx = set(word_idx)
        candidate_pred_positions = []  # 候选预测位置的索引
        for i, ids in enumerate(token_ids):
            # 在遮蔽语言模型任务中不会预测特殊词元，所以如果该位置是特殊词元
            # 那么该位置就不会成为候选mask位置
            if ids in [self.CLS_IDX, self.SEP_IDX]:
                continue
            candidate_pred_positions.append(i)
            # 保存候选位置的索引， 例如可能是 [ 2,3,4,5, ....]
        random.shuffle(candidate_pred_positions)  # 将所有候选位置打乱，更利于后续随机
        # 被掩盖位置的数量，BERT模型中默认将15%的Token进行mask
        num_mlm_preds = max(1, round((len(token_ids) - len(word_idx) + len(word_span_list)) * self.masked_rate))
        # logging.debug(f" ## Mask数量为: {num_mlm_preds}")
        mlm_input_tokens_id, mlm_label = self.replace_masked_tokens(
            token_ids, candidate_pred_positions, num_mlm_preds, word_span_list, word_idx)
        return mlm_input_tokens_id, mlm_label

    def read_csv_split_train_val(self, filepath):
        df = pd.read_csv(filepath)
        df['whole_word'] = df.whole_word.apply(lambda x: eval(x))
        df['ipc'] = df.ipc.apply(lambda x: eval(x))
        df.sample(frac=1).reset_index(drop=True)
        self.train_df = df[:10000]
        self.val_df = df[-4000:]

    def data_process_train(self):
        """
        本函数的作用是是根据格式化后的数据制作NSP和MLM两个任务对应的处理完成的数据
        :param filepath:
        :return:
        """
        # 返回的是一个二维列表，每个列表可以看做是一个段落（其中每个元素为一句话）
        train_data = []
        max_len = 0
        # 这里的max_len用来记录整个数据集中最长序列的长度，在后续可将其作为padding长度的标准
        desc = f" ## 正在构造训练集IPC分类和MLM样本"
        for per_sentence in tqdm(self.train_df.iterrows(), ncols=80, desc=desc):
            sentence = per_sentence[1].content
            whole_word_dict = per_sentence[1].whole_word
            ipc_list = per_sentence[1].ipc
            ipc_label = self.get_ipc_label(ipc_list, self.label_num)
            word_span_list = self.get_word_span_list(whole_word_dict)

            # logging.debug(f" ## 当前句文本：{sentence}")

            token_ids = [self.CLS_IDX] + [self.vocab[token] for token in sentence] + [self.SEP_IDX]
            # logging.debug(f" ## Mask之前词元结果：{[self.vocab.itos[t] for t in token_ids]}")
            segs = [0] * len(token_ids)
            segs = torch.tensor(segs, dtype=torch.long)
            # logging.debug(f" ## Mask之前token ids:{token_ids}")
            # logging.debug(f" ##      segment ids:{segs.tolist()},序列长度为 {len(segs)}")
            mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids, word_span_list)
            token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
            mlm_label = torch.tensor(mlm_label, dtype=torch.long)
            max_len = max(max_len, token_ids.size(0))
            # logging.debug(f" ## Mask之后token ids:{token_ids.tolist()}")
            # logging.debug(f" ## Mask之后词元结果：{[self.vocab.itos[t] for t in token_ids.tolist()]}")
            # logging.debug(f" ## Mask之后label ids:{mlm_label.tolist()}")
            # logging.debug(f" ## 当前样本构造结束================== \n\n")
            train_data.append([token_ids, segs, ipc_label, mlm_label])
        train_all_data = {'data': train_data, 'max_len': max_len}
        return train_all_data

    def data_process_val(self):
        """
        本函数的作用是是根据格式化后的数据制作NSP和MLM两个任务对应的处理完成的数据
        :param filepath:
        :return:
        """
        # 返回的是一个二维列表，每个列表可以看做是一个段落（其中每个元素为一句话）
        val_data = []
        max_len = 0
        # 这里的max_len用来记录整个数据集中最长序列的长度，在后续可将其作为padding长度的标准
        desc = f" ## 正在构造验证集IPC分类和MLM样本"
        for per_sentence in tqdm(self.val_df.iterrows(), ncols=80, desc=desc):
            sentence = per_sentence[1].content
            whole_word_dict = per_sentence[1].whole_word
            ipc_list = per_sentence[1].ipc
            ipc_label = self.get_ipc_label(ipc_list, self.label_num)
            word_span_list = self.get_word_span_list(whole_word_dict)

            # logging.debug(f" ## 当前句文本：{sentence}")

            token_ids = [self.CLS_IDX] + [self.vocab[token] for token in sentence] + [self.SEP_IDX]
            # logging.debug(f" ## Mask之前词元结果：{[self.vocab.itos[t] for t in token_ids]}")
            segs = [0] * len(token_ids)
            segs = torch.tensor(segs, dtype=torch.long)
            # logging.debug(f" ## Mask之前token ids:{token_ids}")
            # logging.debug(f" ##      segment ids:{segs.tolist()},序列长度为 {len(segs)}")
            mlm_input_tokens_id, mlm_label = self.get_masked_sample(token_ids, word_span_list)
            token_ids = torch.tensor(mlm_input_tokens_id, dtype=torch.long)
            mlm_label = torch.tensor(mlm_label, dtype=torch.long)
            max_len = max(max_len, token_ids.size(0))
            # logging.debug(f" ## Mask之后token ids:{token_ids.tolist()}")
            # logging.debug(f" ## Mask之后词元结果：{[self.vocab.itos[t] for t in token_ids.tolist()]}")
            # logging.debug(f" ## Mask之后label ids:{mlm_label.tolist()}")
            # logging.debug(f" ## 当前样本构造结束================== \n\n")
            val_data.append([token_ids, segs, ipc_label, mlm_label])

        val_all_data = {'data': val_data, 'max_len': max_len}
        return val_all_data

    def generate_batch(self, data_batch):
        b_token_ids, b_segs, b_ipc_label, b_mlm_label = [], [], [], []
        for (token_ids, segs, ipc_label, mlm_label) in data_batch:
            # 开始对一个batch中的每一个样本进行处理
            b_token_ids.append(token_ids)
            b_segs.append(segs)
            b_ipc_label.append(ipc_label)
            b_mlm_label.append(mlm_label)
        b_token_ids = pad_sequence(b_token_ids,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=True,
                                   max_len=self.max_sen_len)
        # b_token_ids:  [src_len,batch_size]

        b_segs = pad_sequence(b_segs,  # [batch_size,max_len]
                              padding_value=self.PAD_IDX,
                              batch_first=True,
                              max_len=self.max_sen_len)
        # b_segs: [src_len,batch_size]

        b_mlm_label = pad_sequence(b_mlm_label,  # [batch_size,max_len]
                                   padding_value=self.PAD_IDX,
                                   batch_first=True,
                                   max_len=self.max_sen_len)
        # b_mlm_label:  [src_len,batch_size]

        b_mask = (b_token_ids == self.PAD_IDX)
        # b_mask: [batch_size,max_len]

        b_ipc_label = torch.stack(b_ipc_label)
        # b_nsp_label: [batch_size,label_num]
        return b_token_ids, b_segs, b_mask, b_mlm_label, b_ipc_label

    def load_train_data(self):

        train_all_data = self.data_process_train()
        train_data, max_len = train_all_data['data'], train_all_data['max_len']
        if self.max_sen_len == 'same':
            self.max_sen_len = max_len
        train_iter = DataLoader(train_data, batch_size=self.batch_size,
                                shuffle=self.is_sample_shuffle,
                                collate_fn=self.generate_batch)
        logging.info(f"## 成功返回训练集样本（{len(train_iter.dataset)}）个")
        return train_iter

    def load_val_data(self):

        val_all_data = self.data_process_val()
        val_data = val_all_data['data']
        val_iter = DataLoader(val_data, batch_size=self.batch_size,
                              shuffle=self.is_sample_shuffle,
                              collate_fn=self.generate_batch)
        logging.info(f"## 成功返回验证集样本（{len(val_iter.dataset)}）个")
        return val_iter

