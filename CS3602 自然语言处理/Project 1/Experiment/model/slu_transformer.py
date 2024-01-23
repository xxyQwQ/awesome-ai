import math
import torch
import torch.nn as nn

from utils.lexicon import LexiconMatcher


class PositionalEmbedding(nn.Module):
    def __init__(self, num_features, dropout, max_len=512):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, num_features)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_features, 2).float() * (-math.log(10000.0) / num_features))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, data):
        data = data + self.pe[:, :data.size(1)]
        return self.dropout(data)


class TransformerDecoder(nn.Module):

    def __init__(self, input_size, num_tags, pad_id):
        super(TransformerDecoder, self).__init__()
        self.num_tags = num_tags
        self.output_layer = nn.Linear(input_size, num_tags)
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, hiddens, mask, labels=None):
        logits = self.output_layer(hiddens)
        logits += (1 - mask).unsqueeze(-1).repeat(1, 1, self.num_tags) * -1e32
        prob = torch.softmax(logits, dim=-1)
        if labels is not None:
            loss = self.loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
            return prob, loss
        return (prob, )


class SLUTransformer(nn.Module):

    def __init__(self, config):
        super(SLUTransformer, self).__init__()
        self.config = config
        self.word_embed = nn.Embedding(config.vocab_size, config.embed_size, padding_idx=config.pad_idx)
        self.position_embed = PositionalEmbedding(config.embed_size, config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_size, nhead=config.num_heads, dim_feedforward=config.hidden_size, dropout=config.dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layer)
        self.linear = nn.Linear(config.embed_size, config.hidden_size)
        self.decoder = TransformerDecoder(config.hidden_size, config.num_tags, config.tag_pad_idx)
        self.matcher = LexiconMatcher() if config.refinement else None

    def forward(self, batch, finetune=False):
        tag_ids = batch.tag_ids
        tag_mask = batch.tag_mask
        input_ids = batch.input_ids
        embed = self.word_embed(input_ids)
        embed = self.position_embed(embed)
        transformer_output = self.transformer(embed)
        hidden = torch.tanh(self.linear(transformer_output))
        tag_output = self.decoder(hidden, tag_mask, tag_ids)
        return tag_output

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
