from transformers import AutoModel, AutoTokenizer
from model.common_utils import Node2Vec
import torch
import torch.nn as nn
from tqdm import tqdm



class SciBertNode2Vec(Node2Vec):
    
    def __init__(self, abstract=None, pre_tokenize='./data/pre_tokenize_scibert.pth', pre_embedding='./data/pre_embedding_scibert.pth', device='cuda:0'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        self.bert = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device).to(device)
        if abstract is not None:
            self.abs_ids = self.tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
            self.pre_emb_list = []
            with torch.no_grad():
                for idx in tqdm(range(len(abstract))):
                    emb = self.bert(**self.get_ids_by_idx([idx])).pooler_output
                    self.pre_emb_list.append(emb.detach())
            self.pre_emb_list = torch.cat(self.pre_emb_list, dim=0)
            torch.save(self.abs_ids, pre_tokenize)
            torch.save(self.pre_emb_list, pre_embedding)
        else:
            self.abs_ids = torch.load(pre_tokenize, map_location=device)
            self.pre_emb_list = torch.load(pre_embedding, map_location=device)
            print("Load pre-tokenized data from", pre_tokenize)
            print("Load pre-embedded data from", pre_embedding)
        self.out_trans = nn.Sequential(
            nn.Linear(768, 4096),
            nn.ReLU(),
            nn.Linear(4096, 768)
        ).to(device)
        
    def get_ids_by_idx(self, idx):
        idx_ids = {k:v[idx] for k, v in self.abs_ids.items()}
        return idx_ids
    
    @torch.no_grad()
    def inference(self, abstract):
        inputs = self.tokenizer(abstract, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.bert.device)
        output = self.bert(**inputs).pooler_output
        output = self.out_trans(output)
        # output = torch.cat([output, self.out_trans(output)], dim=1)
        return output

    def forward(self, node_id):
        if type(node_id) == torch.Tensor:
            node_id = node_id.cpu().numpy()
        output = self.pre_emb_list[node_id]
        output = self.out_trans(output)
        # output = torch.cat([output, self.out_trans(output)], dim=1)

        return output



if __name__ == "__main__":
    from dataset.dataloader import load_titleabs
    titleabs = load_titleabs()
    
    abst = titleabs['abs'][:].to_list()
    model = SciBertNode2Vec(abst, device='cuda:0')
    output = model([0, 1, 2, 3, 4])
    print(output.shape)
