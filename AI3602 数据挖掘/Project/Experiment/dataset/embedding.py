from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

@torch.no_grad()
def get_scibert(device="cuda:0"):

    tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
    model = AutoModel.from_pretrained("allenai/scibert_scivocab_uncased").to(device)

    return tokenizer, model

@torch.no_grad()
def encode_single(abstract, tokenizer, model, emb_type="cls", device="cuda:0"):
    """
    emb_type: str, 'cls' or 'mean'. 'cls' returns the pooler_output, 'mean' returns the mean of last_hidden_state
    """

    inputs = tokenizer(
        abstract, return_tensors="pt", truncation=True, padding=True, max_length=512
    ).to(device)
    outputs = model(**inputs)

    if emb_type == "mean":
        embeddings = outputs.last_hidden_state.mean(dim=1)
    elif emb_type == "cls":
        embeddings = outputs.pooler_output

    return embeddings


if __name__ == "__main__":
    
    tokenizer, model = get_scibert()
    from dataset.dataloader import load_titleabs
    titleabs = load_titleabs()
    
    # CLS embeddings
    emb_list = []
    for abs in tqdm(titleabs['abs']):
        emb = encode_single(abs, tokenizer, model)
        emb_list.append(emb)
    embeddings = torch.cat(emb_list, dim=0)
    torch.save(embeddings, './data/embeddings_cls.pth')
    
    # Mean embeddings
    emb_list = []
    for abs in tqdm(titleabs['abs']):
        emb = encode_single(abs, tokenizer, model, emb_type='mean')
        emb_list.append(emb)
    embeddings = torch.cat(emb_list, dim=0)
    torch.save(embeddings, './data/embeddings_mean.pth')
    
