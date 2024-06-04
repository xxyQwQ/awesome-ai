import gradio as gr
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from dataset.dataloader import arxiv_dataset, load_titleabs
from model.bert import BertNode2Vec
from model.scibert import SciBertNode2Vec
from utils.args import get_app_args
from app.get_abstract import title_to_abs
import pandas as pd
import numpy as np



# Initialize arguments and model
args = get_app_args()
device = args.device
data = arxiv_dataset()
titleabs = load_titleabs()

ref_count = np.zeros(data['graph'].num_nodes)
edges = data['graph'].edge_index
srcs, dsts = np.array(edges)
for src, dst in zip(srcs, dsts):
    ref_count[dst] += 1
    
np.max(ref_count), np.min(ref_count), np.mean(ref_count), np.median(ref_count)
# Load pre-trained embeddings
with torch.no_grad():
    if args.model_type == 'bert':
        model = BertNode2Vec(device=device)
        model.load(args.pretrain, device)
        emb = model.embed_all(data)
    elif args.model_type == 'scibert':
        model = SciBertNode2Vec(device=device)
        model.load(args.pretrain, device)
        emb = model.embed_all(data)

F.normalize(emb, p=2, dim=1)

knn = NearestNeighbors(n_neighbors=args.k, metric='cosine')
knn.fit(emb.numpy())

# Define the function for recommendation
def recommend_paper(title_or_abs, title, abstract):
    if not title_or_abs: # 0 for title, 1 for abstract
        try:
            title, abstract = title_to_abs(title)
            with open('gradio.log', 'a') as f:
                f.write(f"Title: {title}, Abstract: {abstract}\n")
        except Exception as e:
            return str(e), []

        if not abstract:
            return "Abstract not found for the given title.", []
    else:
        title = ''

    inf_emb = model.inference(abstract).cpu()
    inf_emb = F.normalize(inf_emb, p=2, dim=1)
    
    distances, nearest_indices = knn.kneighbors(inf_emb.numpy(), return_distance=True)

    sorted_indices_and_distances = sorted(
        zip(nearest_indices[0], distances[0]), 
        key=lambda x: ref_count[x[0]], 
        reverse=True
    )
    
    nearest_indices = [t[0] for t in sorted_indices_and_distances[:args.k]]
    distances = [t[1] for t in sorted_indices_and_distances[:args.k]]
    
    recommendations = []
    for i, idx in enumerate(nearest_indices):
        recommendations.append({
            'Title': titleabs['title'][idx],
            'Abstract': titleabs['abs'][idx],
            'Distance': str(distances[i])
        })
    with open('gradio.log', 'a') as f:
        f.write(f"Title: {title}, Recommendations: {recommendations}\n")
        

    return title, abstract, pd.DataFrame(recommendations)


def recommend_paper_title(title):
    return recommend_paper(0, title, None)

def recommend_paper_abstract(abstract):
    return recommend_paper(1, None, abstract)

if __name__ == "__main__":
    # Define the Gradio interface

    with gr.Blocks() as demo:
        gr.Markdown("# Research Paper Recommendation System")
        gr.Markdown("Enter either the title or abstract of a research paper to get recommendations.")

        with gr.Row():
            with gr.Column():
                title_input = gr.Textbox(lines=1, placeholder="Enter the article title. Example: Attention is can can need.", label="Title", interactive=True)
                recommend_button_title = gr.Button("Submit")
            with gr.Column():
                abstract_input = gr.Textbox(lines=3, placeholder="Write your own abstract to recommend based on it. Example: This paper proposes a new method for ...", label="Abstract", interactive=True)
                recommend_button_abstract = gr.Button("Submit")
        with gr.Row():
            title_box = gr.Textbox(label="Title")
        
        with gr.Row():
            output_recommendations = gr.Dataframe(headers=["Title", "Abstract", "Distance"], wrap=True, height=1200)
            
        title_input.submit(recommend_paper_title, inputs=[title_input], outputs=[title_box, abstract_input, output_recommendations])
        abstract_input.submit(recommend_paper_abstract, inputs=[abstract_input], outputs=[title_box, abstract_input, output_recommendations])
    
        recommend_button_title.click(recommend_paper_title, inputs=[title_input], outputs=[title_box, output_recommendations])
        recommend_button_abstract.click(recommend_paper_abstract, inputs=[abstract_input], outputs=[title_box, output_recommendations])

    demo.launch(share=True, server_port=7998)