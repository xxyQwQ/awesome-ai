from ogb.nodeproppred import Evaluator
from model.common_utils import Node2Vec
from transformers import BertConfig, BertTokenizer, BertModel
import torch.nn as nn
import torch



class BertNode2Vec(Node2Vec):
    
    def __init__(self, abstract=None, pre_tokenize='./data/pre_tokenize_bert.pth', device='cuda:0'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel(BertConfig(num_hidden_layers=1, num_attention_heads=3, intermediate_size=768)).to(device)
        if abstract is not None:
            self.abs_ids = self.tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            torch.save(self.abs_ids, pre_tokenize)
        else:
            self.abs_ids = torch.load(pre_tokenize).to(device)
            print("Load pre-tokenized data from", pre_tokenize)
        
    def get_ids_by_idx(self, idx):
        idx_ids = {k:v[idx] for k, v in self.abs_ids.items()}
        return idx_ids
    
    @torch.no_grad()
    def inference(self, abstract):
        inputs = self.tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.bert.device)
        outputs = self.bert(**inputs)
        return outputs.pooler_output

    def forward(self, node_id):
        if type(node_id) == torch.Tensor:
            node_id = node_id.cpu().numpy()
        inputs = self.get_ids_by_idx(node_id)
        outputs = self.bert(**inputs)

        return outputs.pooler_output



if __name__ == "__main__":
    from dataset.dataloader import load_titleabs
    titleabs = load_titleabs()
    
    abs_10 = titleabs['abs'][:].to_list()
    model = BertNode2Vec(abs_10, device='cuda:0')
    output = model([0, 1, 2, 3, 4])
    print(output.shape)
