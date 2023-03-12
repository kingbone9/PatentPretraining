#!/usr/bin/python
# author kingbone

import os
import logging
import sys

from .utils.log_helper import logger_init
from .models.pretraing_model import BertForPreTraining
from transformers import BertConfig
from .utils.create_pretraining_data import LoadBertPretrainingDataset
from transformers import BertTokenizer
from transformers import AdamW
from transformers import get_polynomial_decay_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from copy import deepcopy
from sklearn import metrics
from sklearn.exceptions import UndefinedMetricWarning
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

import torch
import time


class ModelConfig:
    def __init__(self):
        self.project_dir = os.getcwd()
        self.dataset_dir = os.path.join(self.project_dir, 'data', 'DataFinished_1.csv')
        self.project_dir = os.path.join(self.project_dir, 'mlm+ipc+cl')
        # self.dataset_dir = os.path.join(self.project_dir, 'data', 'debug_reduce.csv')
        self.pretrained_model_dir = "/data/bowen/roberta_wwm/"
        self.data_name = 'patentBert_v1'

        # 如果需要切换数据集，只需要更改上面的配置即可
        self.vocab_path = os.path.join(self.pretrained_model_dir, 'vocab.txt')
        self.device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
        self.model_save_dir = os.path.join(self.project_dir, 'cache')
        self.model_save_path = os.path.join(self.model_save_dir, f'model_split_{self.data_name}')
        self.logs_save_dir = os.path.join(self.project_dir, 'logs')
        self.writer = SummaryWriter(f"runs/{self.data_name}")
        self.is_sample_shuffle = True
        self.use_embedding_weight = True
        self.batch_size = 24
        self.max_sen_len = None  # 为None时则采用每个batch中最长的样本对该batch中的样本进行padding
        self.pad_index = 0
        self.random_state = 2022
        self.learning_rate = 3e-5
        self.weight_decay = 0.1
        self.masked_rate = 0.15
        self.masked_token_rate = 0.8
        self.masked_token_unchanged_rate = 0.5
        self.whole_word_masked_rate = 0.5
        self.log_level = logging.DEBUG
        self.use_torch_multi_head = False  # False表示使用model/BasicBert/MyTransformer中的多头实现
        self.epochs = 3
        self.model_val_per_epoch = 1

        logger_init(log_file_name=self.data_name, log_level=self.log_level,
                    log_dir=self.logs_save_dir)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)
        self.bert_config = BertConfig.from_pretrained(self.pretrained_model_dir)
        self.bert_config.ipc_num = 125
        for key, value in self.bert_config.__dict__.items():
            self.__dict__[key] = value
        # 将当前配置打印到日志文件中
        logging.info(" ### 将当前配置打印到日志文件中 ")
        for key, value in self.__dict__.items():
            logging.info(f"### {key} = {value}")


def train(config):
    model = BertForPreTraining.from_pretrained(config.pretrained_model_dir, config=config.bert_config)
    last_epoch = -1
    if os.path.exists(config.model_save_path):
        model = BertForPreTraining.from_pretrained(config.model_save_path, config=config.bert_config)

        logging.info("## 成功载入已有模型，进行追加训练......")
    model = model.to(config.device)
    model.train()
    bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir)
    data_loader = LoadBertPretrainingDataset(vocab_path=config.vocab_path,
                                             tokenizer=bert_tokenize,
                                             batch_size=config.batch_size,
                                             max_sen_len=config.max_sen_len,
                                             max_position_embeddings=config.max_position_embeddings,
                                             pad_index=config.pad_index,
                                             is_sample_shuffle=config.is_sample_shuffle,
                                             random_state=config.random_state,
                                             data_name=config.data_name,
                                             masked_rate=config.masked_rate,
                                             masked_token_rate=config.masked_token_rate,
                                             masked_token_unchanged_rate=config.masked_token_unchanged_rate,
                                             whole_word_masked_rate=config.whole_word_masked_rate)
    data_loader.read_csv_split_train_val(config.dataset_dir)
    val_iter = data_loader.load_val_data()
    # train_iter = data_loader.load_train_val_data(config.dataset_dir)
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay,
            "initial_lr": config.learning_rate

        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "initial_lr": config.learning_rate
        },
    ]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = AdamW(optimizer_grouped_parameters)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                          int(138404 * 0),
                                                          int(config.epochs * 138404),
                                                          last_epoch=last_epoch)
    max_acc = 0
    state_dict = None
    for epoch in range(config.epochs):

        train_iter = data_loader.load_train_data()
        losses = 0
        start_time = time.time()
        for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_ipc_label) in enumerate(train_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            b_ipc_label = b_ipc_label.to(config.device)
            loss_list, mlm_logits, ipc_logits = model(input_ids=b_token_ids,
                                                      attention_mask=b_mask,
                                                      token_type_ids=b_segs,
                                                      labels=b_mlm_label,
                                                      ipc_labels=b_ipc_label,
                                                      )[:3]

            t_loss, mlm_loss, ipc_loss = loss_list
            if epoch == 0 and idx < 400000:
                loss = mlm_loss
            else:
                loss = t_loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            losses += loss.item()

            if idx % 100 == 0:
                mlm_acc, _, _ = accuracy_mlm(mlm_logits, b_mlm_label, data_loader.PAD_IDX)
                f1_macro, f1_micro = accuracy_ipc(ipc_logits.cpu().detach().numpy(), b_ipc_label.cpu().detach().numpy())
                logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Batch[{idx}/{len(train_iter)}], "
                             f"Train loss :{loss.item():.3f}, Train mlm loss: {mlm_loss.item():.3f},"
                             f"Train ipc loss :{ipc_loss.item():.3f}, Train mlm acc: {mlm_acc:.3f},"
                             f"ipc prediction f1 macro: {f1_macro:.2f}, ipc prediction f1 micro: {f1_micro:.2f}")
                config.writer.add_scalar('Training/Loss', loss.item(), scheduler.last_epoch)
                config.writer.add_scalar('Training/Learning Rate', scheduler.get_last_lr()[0], scheduler.last_epoch)
                config.writer.add_scalars(main_tag='Training/Accuracy',
                                          tag_scalar_dict={'IPC PREDICTION F1 MACRO': f1_macro,
                                                           'IPC PREDICTION F1 MICRO': f1_micro,
                                                           'MLM': mlm_acc},
                                          global_step=scheduler.last_epoch)

            if idx % 1000 == 0:
                eval_losses, eval_mlm_losses, eval_ipc_losses, eval_mlm_acc, eval_f1_macro, eval_f1_micro = evaluate(
                    config, val_iter, model, data_loader.PAD_IDX)
                logging.info(f"Eval total loss :{eval_losses:.3f}, Eval mlm loss: {eval_mlm_losses:.3f},"
                             f"Eval ipc loss :{eval_ipc_losses:.3f}, Eval mlm acc: {eval_mlm_acc:.3f},"
                             f"ipc prediction f1 macro: {eval_f1_macro:.2f}, ipc prediction f1 micro: {eval_f1_micro:.2f}")
                if eval_mlm_acc > max_acc:
                    max_acc = eval_mlm_acc
                    model.save_pretrained(config.model_save_path)

        end_time = time.time()
        train_loss = losses / len(train_iter)
        logging.info(f"Epoch: [{epoch + 1}/{config.epochs}], Train loss: "
                     f"{train_loss:.3f}, Epoch time = {(end_time - start_time):.3f}s")


def prob2zeroone(label_prob, thre, num_label=125):
    thre_filer = label_prob > thre
    if thre_filer.any():
        return thre_filer.astype(int).tolist()
    else:
        label_idx = np.zeros(num_label, dtype=int)
        return label_idx.tolist()


def clr(labels, preds, thre):
    preds_get = []
    for pred in preds:
        pred = prob2zeroone(pred, thre)
        preds_get.append(pred)
    preds_get = np.array(preds_get)
    return metrics.classification_report(labels, preds_get, output_dict=True)


def accuracy_mlm(mlm_logits, mlm_labels, PAD_IDX):
    """
    :param mlm_logits:  [src_len,batch_size,src_vocab_size]
    :param mlm_labels:  [src_len,batch_size]
    :param nsp_logits:  [batch_size,2]
    :param nsp_label:  [batch_size]
    :param PAD_IDX:
    :return:
    """
    mlm_pred = mlm_logits.argmax(axis=2).reshape(-1)
    mlm_true = mlm_labels.reshape(-1)
    mlm_acc = mlm_pred.eq(mlm_true)  # 计算预测值与正确值比较的情况
    mask = torch.logical_not(mlm_true.eq(PAD_IDX))  # 找到真实标签中，mask位置的信息。 mask位置为FALSE，非mask位置为TRUE
    mlm_acc = mlm_acc.logical_and(mask)  # 去掉acc中mask的部分
    mlm_correct = mlm_acc.sum().item()
    mlm_total = mask.sum().item()
    mlm_acc = float(mlm_correct) / mlm_total

    return [mlm_acc, mlm_correct, mlm_total]


def accuracy_ipc(ipc_logits, ipc_label):
    """
    :param mlm_logits:  [src_len,batch_size,src_vocab_size]
    :param mlm_labels:  [src_len,batch_size]
    :param nsp_logits:  [batch_size,2]
    :param nsp_label:  [batch_size]
    :param PAD_IDX:
    :return:
    """

    step_logits = []
    step_labels = []

    [step_logits.append(logit.tolist()) for logit in ipc_logits]
    [step_labels.append(label.tolist()) for label in ipc_label]

    clr_step = clr(np.array(step_labels), np.array(step_logits), 0.2)

    f1_macro, f1_micro = clr_step['macro avg']['f1-score'], clr_step['micro avg']['f1-score']

    return [f1_macro, f1_micro]


def evaluate(config, data_iter, model, PAD_IDX):
    model.eval()
    all_ipc_logits = torch.tensor([]).to(config.device)
    all_ipc_labels = torch.tensor([]).to(config.device)
    mlm_corrects, mlm_totals = 0, 0
    losses = 0
    t_mlm_losses = 0
    t_ipc_losses = 0
    with torch.no_grad():
        for idx, (b_token_ids, b_segs, b_mask, b_mlm_label, b_ipc_label) in enumerate(data_iter):
            b_token_ids = b_token_ids.to(config.device)  # [src_len, batch_size]
            b_segs = b_segs.to(config.device)
            b_mask = b_mask.to(config.device)
            b_mlm_label = b_mlm_label.to(config.device)
            b_ipc_label = b_ipc_label.to(config.device)
            t_loss, mlm_logits, ipc_logits = model(input_ids=b_token_ids,
                                                   attention_mask=b_mask,
                                                   token_type_ids=b_segs,
                                                   labels=b_mlm_label,
                                                   ipc_labels=b_ipc_label,
                                                   )[:3]
            loss, mlm_loss, ipc_loss = t_loss
            losses += loss.item()
            t_mlm_losses += mlm_loss.item()
            t_ipc_losses += ipc_loss.item()

            mlm_acc, mlm_cor, mlm_tot = accuracy_mlm(mlm_logits, b_mlm_label, PAD_IDX)
            mlm_corrects += mlm_cor
            mlm_totals += mlm_tot

            all_ipc_logits = torch.cat((all_ipc_logits, ipc_logits))
            all_ipc_labels = torch.cat((all_ipc_labels, b_ipc_label))
    model.train()
    losses /= len(data_iter)
    t_mlm_losses /= len(data_iter)
    t_ipc_losses /= len(data_iter)
    f1_macro, f1_micro = accuracy_ipc(all_ipc_logits.cpu().detach().numpy(), all_ipc_labels.cpu().detach().numpy())

    return losses, t_mlm_losses, t_ipc_losses, float(mlm_corrects) / mlm_totals, f1_macro, f1_micro


# def inference(config, sentences=None, masked=False, language='en', random_state=None):
#     bert_tokenize = BertTokenizer.from_pretrained(config.pretrained_model_dir).tokenize
#     data_loader = LoadBertPretrainingDataset(vocab_path=config.vocab_path,
#                                              tokenizer=bert_tokenize,
#                                              pad_index=config.pad_index,
#                                              random_state=config.random_state,
#                                              masked_rate=0.15)  # 15% Mask掉
#     token_ids, pred_idx, mask = data_loader.make_inference_samples(sentences,
#                                                                    masked=masked,
#                                                                    language=language,
#                                                                    random_state=random_state)
#     model = BertForPretrainingModel(config,
#                                     config.pretrained_model_dir)
#     if os.path.exists(config.model_save_path):
#         checkpoint = torch.load(config.model_save_path)
#         loaded_paras = checkpoint['model_state_dict']
#         model.load_state_dict(loaded_paras)
#         logging.info("## 成功载入已有模型进行推理......")
#     else:
#         raise ValueError(f"模型 {config.model_save_path} 不存在！")
#     model = model.to(config.device)
#     model.eval()
#     with torch.no_grad():
#         token_ids = token_ids.to(config.device)  # [src_len, batch_size]
#         mask = mask.to(config.device)
#         mlm_logits, _ = model(input_ids=token_ids,
#                               attention_mask=mask)
#     pretty_print(token_ids, mlm_logits, pred_idx,
#                  data_loader.vocab.itos, sentences, language)


def pretty_print(token_ids, logits, pred_idx, itos, sentences, language):
    """
    格式化输出结果
    :param token_ids:   [src_len, batch_size]
    :param logits:  [src_len, batch_size, vocab_size]
    :param pred_idx:   二维列表，每个内层列表记录了原始句子中被mask的位置
    :param itos:
    :param sentences: 原始句子
    :return:
    """
    token_ids = token_ids.transpose(0, 1)  # [batch_size,src_len]
    logits = logits.transpose(0, 1)  # [batch_size, src_len,vocab_size]
    y_pred = logits.argmax(axis=2)  # [batch_size, src_len]
    sep = " " if language == 'en' else ""
    for token_id, sentence, y, y_idx in zip(token_ids, sentences, y_pred, pred_idx):
        sen = [itos[id] for id in token_id]
        sen_mask = sep.join(sen).replace(" ##", "").replace("[PAD]", "").replace(" ,", ",")
        sen_mask = sen_mask.replace(" .", ".").replace("[SEP]", "").replace("[CLS]", "").lstrip()
        logging.info(f" ### 原始: {sentence}")
        logging.info(f"  ## 掩盖: {sen_mask}")
        for idx in y_idx:
            sen[idx] = itos[y[idx]].replace("##", "")
        sen = sep.join(sen).replace("[PAD]", "").replace(" ,", ",")
        sen = sen.replace(" .", ".").replace("[SEP]", "").replace("[CLS]", "").lstrip()
        logging.info(f"  ## 预测: {sen}")
        logging.info("===============")


if __name__ == '__main__':
    config = ModelConfig()
    train(config)
