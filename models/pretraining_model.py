#!/usr/bin/python
# author kingbone
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Optional, Tuple
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertLMPredictionHead
from transformers.data.data_collator import DataCollatorForLanguageModeling

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"


@dataclass
class BertForMLMIPCPredictOutput(ModelOutput):
    """
    Output type of :class:`~transformers.BertForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    ipc_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MLPLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


# 相似度层，计算向量间相似度
class Similarity(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.temp = 0.05
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class IPCPredictLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.out_mesh_dstrbtn = nn.Linear(config.hidden_size, config.ipc_num)
        nn.init.xavier_uniform_(self.out_mesh_dstrbtn.weight)

    def forward(self, context_vectors):
        output_dstrbtn = self.out_mesh_dstrbtn(context_vectors)  # (batch, n_labels)
        output_dstrbtn = output_dstrbtn
        return output_dstrbtn


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.masked_token_predictions = BertLMPredictionHead(config)
        self.ipc_predictions = IPCPredictLayer(config)
        self.contrastive = Similarity(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, sequence_output, pooled_output, contrastive_output):
        cos_sim = self.contrastive(pooled_output.unsqueeze(1),contrastive_output.unsqueeze(0))
        pooled_output = self.dropout(pooled_output)
        prediction_scores = self.masked_token_predictions(sequence_output)
        ipc_logits = self.ipc_predictions(pooled_output)
        return prediction_scores, ipc_logits, cos_sim


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "cls_before_pooler", "avg", "avg_top2",
                                    "avg_first_last"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_states = outputs.hidden_states

        if self.pooler_type in ['cls_before_pooler', 'cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class BertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            ipc_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape ``(batch_size, sequence_length)``, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        next_sentence_label (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair
            (see :obj:`input_ids` docstring) Indices should be in ``[0, 1]``:

            - 0 indicates sequence B is a continuation of sequence A,
            - 1 indicates sequence B is a random sequence.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:
        """
        device = input_ids.device
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        contrastive_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        _, constrastive_pooled_output = contrastive_outputs[:2]
        prediction_scores, ipc_logits, cos_sim = self.cls(sequence_output, pooled_output, constrastive_pooled_output)

        contrastive_label = torch.arange(cos_sim.size(0)).long().to(device)
        total_loss = None
        if labels is not None and ipc_labels is not None:
            loss_fct_mlm = CrossEntropyLoss(ignore_index=0)
            loss_fct_ce = CrossEntropyLoss()
            loss_fct_bce = BCEWithLogitsLoss()
            masked_lm_loss = loss_fct_mlm(prediction_scores.reshape(-1, self.config.vocab_size), labels.reshape(-1))
            ipc_predict_loss = loss_fct_bce(ipc_logits, ipc_labels)
            contrastive_loss = loss_fct_ce(cos_sim, contrastive_label)
            total_loss = masked_lm_loss + ipc_predict_loss + contrastive_loss

        if not return_dict:
            output = (prediction_scores, ipc_logits) + outputs[2:]
            return ((total_loss, masked_lm_loss, ipc_predict_loss,) + output) if total_loss is not None else output

        return BertForMLMIPCPredictOutput(
            loss=[total_loss, masked_lm_loss, ipc_predict_loss],
            prediction_logits=prediction_scores,
            ipc_logits=ipc_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
