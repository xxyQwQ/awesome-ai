from transformers import DistilBertConfig, DistilBertTokenizer, DistilBertModel
from model.common_utils import Node2Vec
import torch.nn as nn
import torch



class DistilBertNode2Vec(Node2Vec):
    
    def __init__(self, abstract=None, pre_tokenize='./data/pre_tokenize_distilbert.pth', device='cuda:0'):
        super().__init__()
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert = DistilBertModel(DistilBertConfig(n_layers=1, n_heads=2, hidden_dim=768)).to(device)
        if abstract is not None:
            self.abs_ids = self.tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            torch.save(self.abs_ids, pre_tokenize)
        else:
            self.abs_ids = torch.load(pre_tokenize, map_location=device)
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

        return outputs.pooler_output  # Use the first token (CLS token) as the pooled output



if __name__ == "__main__":
    from dataset.dataloader import load_titleabs
    titleabs = load_titleabs()
    
    abst = titleabs['abs'][:].to_list()
    model = DistilBertNode2Vec(abst, device='cuda:0')
    output = model([0, 1, 2, 3, 4])
    print(output.shape)
