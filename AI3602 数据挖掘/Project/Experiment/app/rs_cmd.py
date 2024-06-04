from dataset.dataloader import arxiv_dataset, load_titleabs
import torch
from model.bert import BertNode2Vec
from model.scibert import SciBertNode2Vec
from utils.args import get_app_args 
import torch.nn.functional as F
from app.get_abstract import title_to_abs
import numpy as np
from sklearn.neighbors import NearestNeighbors



args = get_app_args()

device = args.device
data = arxiv_dataset()
titleabs = load_titleabs()

ref_count = np.zeros(data['graph'].num_nodes)
edges = data['graph'].edge_index
srcs, dsts = np.array(edges)
for src, dst in zip(srcs, dsts):
    ref_count[dst] += 1
    
with torch.no_grad():
    # emb = torch.load('./data/embeddings_cls.pth').cpu()
    if args.model_type == 'bert':
        model = BertNode2Vec(device=device)
        model.load(args.pretrain, device)
        emb = model.embed_all(data)
    elif args.model_type == 'scibert':
        model = SciBertNode2Vec(device=device)
        model.load(args.pretrain, device)
        emb = model.embed_all(data)
        
F.normalize(emb, p=2, dim=1)

knn = NearestNeighbors(n_neighbors=args.k*3, metric='cosine')
knn.fit(emb.numpy())

while True:
    try:
        title, abstract = title_to_abs(input("Enter the article title: "))
    except Exception as e:
        print(e, '\nTry again.')
        continue

    if not abstract:
        continue
    else:
        inf_emb = model.inference(abstract).cpu()
        # tokenizer, model = get_scibert()
        # inf_emb = encode_single(abstract, tokenizer, model).cpu()
        inf_emb = F.normalize(inf_emb, p=2, dim=1)
        
        distances, nearest_indices = knn.kneighbors(inf_emb.numpy(), return_distance=True)

        sorted_indices_and_distances = sorted(
            zip(nearest_indices[0], distances[0]), 
            key=lambda x: ref_count[x[0]], 
            reverse=True
        )
        
        print(sorted_indices_and_distances)
        nearest_indices = [t[0] for t in sorted_indices_and_distances[:args.k]]
        distances = [t[1] for t in sorted_indices_and_distances[:args.k]]

        print(f"The indices of the {args.k} nearest nodes are: {nearest_indices[0]}")
        print(f"The distances are: {distances[0]}")
        
        print('\n\nRecommendation:\n')
        print('-----------------')
        for idx in nearest_indices:
            print(titleabs['title'][idx])
            print(titleabs['abs'][idx])
            print('-----------------')
        print('\n\n')