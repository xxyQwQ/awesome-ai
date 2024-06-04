from dataset.dataloader import arxiv_dataset, ClassifierDataset
from model.common_utils import Classifier
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.common_utils import evaluate
from model.bert import BertNode2Vec
from model.scibert import SciBertNode2Vec
from model.embedding import Embedding
from model.hashmlp import MLP
from model.transformer import TransformerNode2Vec
from tqdm import tqdm
from utils.args import get_vaildate_args
from sklearn.neighbors import KNeighborsClassifier
import torch.nn.functional as F



if __name__ == "__main__":
    args = get_vaildate_args()
    
    device = args.device
    
    data = arxiv_dataset()
    label_train = data['graph'].y[np.concatenate([data['train_idx'], data['valid_idx']])]
    label_test = data['graph'].y[data['test_idx']]
    
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
    
    emb = F.normalize(emb, p=2, dim=1).detach()
    # emb_np = emb.cpu().numpy()
    # np.save('emb.npy', emb_np)
    emb_train = emb[np.concatenate([data['train_idx'], data['valid_idx']])]
    emb_test = emb[data['test_idx']]
    
    if args.classifier == 'mlp':
    
        classifier = Classifier(in_dim=emb.shape[1], num_cls=40).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(classifier.parameters(), lr=args.lr)
        
        train_loader = DataLoader(ClassifierDataset(emb_train, label_train), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, pin_memory_device=device)
        test_loader = DataLoader(ClassifierDataset(emb_test, label_test), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, pin_memory_device=device)
        
        
        for epoch in range(args.num_epochs):
            classifier.train()
            total_loss = 0

            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()

                optimizer.zero_grad()
                outputs = classifier(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch [{epoch+1}/{args.num_epochs}], avg_loss: {avg_loss:.4f}")

    elif args.classifier == 'knn':
        
        classifier = KNeighborsClassifier(n_neighbors=args.k)
        # classifier = KNeighborsClassifier(n_neighbors=args.k, metric='cosine')
        classifier.fit(emb_train, label_train.reshape(-1))
        

    print("Training completed.")
    
    print("Evaluating...")
    if args.classifier == 'mlp':
    
        classifier.eval()
        with torch.no_grad():
            pred = []
            true = []
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).squeeze()
                outputs = classifier(batch_x)
                pred.append(outputs.argmax(dim=1))
                true.append(batch_y)
            pred = torch.cat(pred).cpu().numpy().reshape(-1, 1)
            true = torch.cat(true).cpu().numpy().reshape(-1, 1)
        
    elif args.classifier == 'knn':
        pred = classifier.predict(emb_test).reshape(-1, 1)
        true = label_test.reshape(-1, 1)
        
    result = evaluate(pred, true)
    print('Accuracy:', result)