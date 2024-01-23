import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from torchcrf import CRF

from utils.lexicon import LexiconMatcher


class BertRNNCRFDecoder(nn.Module):

    def __init__(self, num_dims, num_tags):
        super(BertRNNCRFDecoder, self).__init__()
        self.linear = nn.Linear(num_dims, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, hiddens, mask, labels=None):
        logits = self.linear(hiddens)
        pred = self.crf.decode(logits, mask.bool())
        if labels is not None:
            loss = -self.crf(logits, labels, mask.bool(), reduction='mean')
            return pred, loss
        return (pred, )


class SLUBertRNNCRF(nn.Module):

    def __init__(self, config):
        super(SLUBertRNNCRF, self).__init__()
        self.config = config
        self.device = config.device_name
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_version)
        self.bert = AutoModel.from_pretrained(config.bert_version)
        self.rnn = getattr(nn, config.encoder_cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(p=config.dropout)
        self.decoder = BertRNNCRFDecoder(config.hidden_size, config.num_tags)
        self.matcher = LexiconMatcher() if config.refinement else None

    def forward(self, batch, finetune=False):
        sentences = [' '.join(sentence.replace(' ', '-')) for sentence in batch.utt] # force to split words
        inputs = self.tokenizer(sentences, padding=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        if finetune:
            self.bert.train()
            hidden_state = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            self.bert.eval()
            with torch.no_grad():
                hidden_state = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        hidden_state = hidden_state[:, 1:-1, :]
        rnn_outputs, _ = self.rnn(hidden_state)
        rnn_outputs = self.dropout(rnn_outputs)
        output_tags = self.decoder(rnn_outputs, batch.tag_mask, batch.tag_ids)
        return output_tags

    def decode(self, label_vocab, batch):
        batch_size = len(batch)
        labels = batch.labels
        output = self.forward(batch)
        preds = output[0]
        predictions = []
        for i in range(batch_size):
            pred = preds[i]
            pred_tuple = []
            idx_buff, tag_buff, pred_tags = [], [], []
            pred = pred[:len(batch.utt[i])]
            for idx, tid in enumerate(pred):
                tag = label_vocab.convert_idx_to_tag(tid)
                pred_tags.append(tag)
                if (tag == 'O' or tag.startswith('B')) and len(tag_buff) > 0:
                    slot = '-'.join(tag_buff[0].split('-')[1:])
                    value = ''.join([batch.utt[i][j] for j in idx_buff])
                    if self.matcher is not None:
                        value = self.matcher.match(slot, value)
                    idx_buff, tag_buff = [], []
                    pred_tuple.append(f'{slot}-{value}')
                    if tag.startswith('B'):
                        idx_buff.append(idx)
                        tag_buff.append(tag)
                elif tag.startswith('I') or tag.startswith('B'):
                    idx_buff.append(idx)
                    tag_buff.append(tag)
            if len(tag_buff) > 0:
                slot = '-'.join(tag_buff[0].split('-')[1:])
                value = ''.join([batch.utt[i][j] for j in idx_buff])
                if self.matcher is not None:
                    value = self.matcher.match(slot, value)
                pred_tuple.append(f'{slot}-{value}')
            predictions.append(pred_tuple)
        if len(output) == 1:
            return predictions
        else:
            loss = output[1]
            return predictions, labels, loss.cpu().item()
