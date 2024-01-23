import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification
from torchcrf import CRF

from utils.lexicon import LexiconMatcher


class BertCRFDecoder(nn.Module):

    def __init__(self, num_tags):
        super(BertCRFDecoder, self).__init__()
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, logits, mask, labels=None):
        pred = self.crf.decode(logits, mask.bool())
        if labels is not None:
            loss = -self.crf(logits, labels, mask.bool(), reduction='mean')
            return pred, loss
        return (pred, )


class SLUBertCRF(nn.Module):

    def __init__(self, config):
        super(SLUBertCRF, self).__init__()
        self.config = config
        self.device = config.device_name
        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_version)
        self.bert = AutoModelForTokenClassification.from_pretrained(config.bert_version, num_labels=config.num_tags)
        self.decoder = BertCRFDecoder(config.num_tags)
        self.matcher = LexiconMatcher() if config.refinement else None

    def forward(self, batch, finetune=False):
        sentences = [' '.join(sentence.replace(' ', '-')) for sentence in batch.utt] # force to split words
        inputs = self.tokenizer(sentences, padding=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        if finetune:
            outputs = self.bert(input_ids, attention_mask=attention_mask).logits[:, 1:-1, :]
        else:
            with torch.no_grad():
                outputs = self.bert(input_ids, attention_mask=attention_mask).logits[:, 1:-1, :]
        output_tags = self.decoder(outputs, batch.tag_mask, batch.tag_ids)
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
