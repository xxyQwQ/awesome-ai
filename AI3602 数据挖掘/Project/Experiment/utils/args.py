import argparse
import random
import numpy as np
import torch
from time import strftime
import os
import json



def get_train_args():
    parser = argparse.ArgumentParser(description='Train a BertNode2Vec model.')

    # Trainer arguments
    parser.add_argument('--model_type', type=str, default='bert', help='Path to the pre-trained model.')
    parser.add_argument('--n_negs', type=int, default=5, help='Number of negative samples to be used in negative sampling.')
    parser.add_argument('--n_epochs', type=int, default=100, help='Number of epochs to train the model.')
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate for training.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run the training.')
    parser.add_argument('--num_workers', type=int, default=20, help='Number of workers for parallel processing.')
    parser.add_argument('--walk_length', type=int, default=6, help='Length of each random walk session.')
    parser.add_argument('--window_size', type=int, default=5, help='Window size for each training sample.')
    parser.add_argument('--n_walks_per_node', type=int, default=3, help='Number of walks to start from each node.')
    parser.add_argument('--sample_node_prob', type=float, default=0.1, help='Probability of sampling a node.')
    
    parser.add_argument('--pretrain', type=str, default=None, help='Path to the pre-trained model.')
    
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')

    args = parser.parse_args()
    seed_everything(args.seed)
    args.save_path = make_save_dir(args)

    return args


def get_vaildate_args():
    parser = argparse.ArgumentParser(description='Validate model or baseline.')

    # classifier train
    parser.add_argument('--classifier', type=str, default='mlp', help=' Type of the classifier.')
    
    # MLP classifier
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the classifier.')
    parser.add_argument('--batch_size', type=int, default=int(2**10), help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run.')
    parser.add_argument('--num_workers', type=int, default=24, help='Number of workers for parallel processing.')
    
    # KNN classifier
    parser.add_argument('--k', type=int, default=100, help='Number of neighbors to consider in KNN.')
    
    # validate options
    parser.add_argument('--model_type', type=str, default='scibert', help='Type of model to validate.')
    parser.add_argument('--pretrain', type=str, default=None, help='Path to the pre-trained model.')
    
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    
    args = parser.parse_args()
    seed_everything(args.seed)

    return args


def get_app_args():
    parser = argparse.ArgumentParser(description='Validate model or baseline.')

    parser.add_argument('--k', type=int, default=5, help='Number of neighbors to recommend in recommendation system.')
    
    parser.add_argument('--model_type', type=str, default='scibert', help='Type of model to validate.')
    parser.add_argument('--pretrain', type=str, default='./checkpoint/test.pth', help='Path to the pre-trained model.')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility.')
    
    args = parser.parse_args()
    seed_everything(args.seed)

    return args


def seed_everything(seed):
    """
    Seed all random number generators for reproducibility.
    
    Args:
    - seed (int): The seed value to use.
    """
    # Seed Python random module
    random.seed(seed)
    
    # Seed NumPy
    np.random.seed(seed)
    
    # Seed PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    
    # For CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Enable benchmark mode for optimal performance


def make_save_dir(args):
    save_path = f'./checkpoint/{strftime("%Y%m%d%H%M%S")}'
    os.makedirs(save_path)
    with open(os.path.join(save_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f)
    return save_path