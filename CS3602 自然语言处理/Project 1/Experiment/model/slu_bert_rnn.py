import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from utils.lexicon import LexiconMatcher


class BertRNNDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(BertRNNDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.softmax_fn = nn.Softmax(dim=-1)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, masks, labels=None):
        logits = self.output_layer(hiddens)
        logits += -1e32 * (1 - masks).unsqueeze(-1).repeat(1, 1, self.num_tags)
        prob = self.softmax_fn(logits)
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        else:
            return (prob, )


class SLUBertRNN(nn.Module):

    def __init__(self, config):
        super(SLUBertRNN, self).__init__()
        self.config = config
        self.device = config.device_name
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_version)
        self.bert = AutoModel.from_pretrained(config.bert_version)
        self.rnn = getattr(nn, config.encoder_cell)(config.embed_size, config.hidden_size // 2, num_layers=config.num_layer, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(config.dropout)
        self.decoder = BertRNNDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)
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
        prob = output[0]
        predictions = []
        for i in range(batch_size):
            pred = torch.argmax(prob[i], dim=-1).cpu().tolist()
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
