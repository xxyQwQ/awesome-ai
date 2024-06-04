from dataset.dataloader import arxiv_dataset, get_test_edges
from model.common_utils import calc_auc_score
import torch
from model.bert import BertNode2Vec
from model.scibert import SciBertNode2Vec
from model.embedding import Embedding
from model.hashmlp import MLP
from model.transformer import TransformerNode2Vec
from utils.args import get_vaildate_args
import torch.nn.functional as F
from tqdm import tqdm



if __name__ == "__main__":
    args = get_vaildate_args()
    
    device = args.device
    
    data = arxiv_dataset()
    test_edge, test_label = get_test_edges(data, year=[2020])
    
    with torch.no_grad():
        if args.model_type == 'random':
            emb = torch.rand((data['graph'].num_nodes, 768))
            
        elif args.model_type == 'randombert':
            model = BertNode2Vec(device=device)
            emb = model.embed_all(data)
            
        elif args.model_type == 'embedding':
            model = Embedding(num_node=data['graph'].num_nodes, embedding_dim=768, device='cpu') # don't cuda
            model.load(args.pretrain, 'cpu')
            emb = model.embedding.weight.detach()
            
        elif args.model_type == 'mlp':
            model = MLP(in_dim=132, embedding_dim=768, hidden=16384, device=args.device)
            model.load(args.pretrain, args.device)
            emb = model.embed_all(data)
                        
        elif args.model_type == 'scibert_direct':
            emb = torch.load('./data/embeddings_cls.pth').cpu()
            
        elif args.model_type == 'pretrained_bert':
            model = SciBertNode2Vec(device=device)
            model.load(args.pretrain, device)
            emb = model.embed_all(data)
                
        elif args.model_type == 'bert':
            model = BertNode2Vec(device=device)
            model.load(args.pretrain, device)
            emb = model.embed_all(data)
    
        elif args.model_type == 'transformer':
            model = TransformerNode2Vec(device=device)
            model.load(args.pretrain, device)
            emb = model.embed_all(data)
    
    emb = F.normalize(emb, p=2, dim=1).to(device)
    
    predictions = []
    for edge, label in tqdm(zip(test_edge, test_label)):
        src, dst = edge
        pred = torch.nn.Sigmoid()(torch.matmul(emb[src].unsqueeze(0), emb[dst].unsqueeze(0).t())).item()
        predictions.append(pred)
    
    result = calc_auc_score(predictions, test_label)
    print('AUC:', result)