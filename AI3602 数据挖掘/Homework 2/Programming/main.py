import os
import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from models.resnet import resnet8
from utils import get_model_param_vec, get_model_grad_vec, update_grad, accuracy, AverageMeter


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Hyper-parameters
workers = 4
epochs = 30
batch_size = 128
weight_decay = 1e-4
print_freq = 50
trajectory_dir = "./trajectory"
n_components = 15
params_number = 30
lr = 0.1


class PCA:
    '''
        Implement PCA using svd
        Numpy only: (np.linalg.svd is suggested)
    '''
    def __init__(self, n_components):
        '''_summary_

        Args:
            n_components (int): Number of components to keep

        self.components_ (ndarray): Principal axes in feature space. 
            shape: (n_components, n_features)
        self.explained_variance_ (ndarray): The amount of variance explained by each of the selected components. 
            shape: (n_components)
        self.explained_variance_ratio_ (ndarray): Percentage of variance explained by each of the selected components.
            shape: (n_components)
        '''
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        '''fit PCA for the given matrix X.
        Results should be stored in: 
            self.components_
            self.explained_variance_
            self.explained_variance_ratio_

        Args:
            X (ndarray): shape: (n_samples, n_features)

        Return:
        '''
        ## TODO
        n_samples, _ = X.shape
        X_normalized = X - np.mean(X, axis=0)
        _, scalers, components = np.linalg.svd(X_normalized, full_matrices=False)
        explained_variance = np.square(scalers) / (n_samples - 1)
        explained_variance_ratio = explained_variance / np.sum(explained_variance)
        self.components_ = components[:self.n_components]
        self.explained_variance_ = explained_variance[:self.n_components]
        self.explained_variance_ratio_ = explained_variance_ratio[:self.n_components]
        ## End of TODO


def P_SGD(model, optimizer, grad, P):
    ''' Special SGD for this project. Your grad will be projected to
    low dimensional subspace with P and then projected back to parameter space.
    (It sounds wierd here, but you can imagine that in some condition 
    we can only transmit several bits) 

    Args:
        model   
        optimizer 
        grad    
        P : project matrix
    '''
    # project the gradients into low-dimentional subspace
    gk = torch.mm(P, grad.reshape(-1, 1))

    assert gk.shape[0] == n_components

    # reproject the gradients into model parameter space
    grad_proj = torch.mm(P.transpose(0, 1), gk)

    # Update the model grad and do a step
    update_grad(model, grad_proj)
    optimizer.step()


def train(train_loader, model, criterion, optimizer, epoch, P):
    # Run one train epoch

    # Switch to train mode
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    for i, (input, target) in enumerate(train_loader):
        # Load batch data to cuda
        target = target.to(device)
        input_var = input.to(device)
        target_var = target

        # Compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Do P_SGD
        gk = get_model_grad_vec(model)
        P_SGD(model, optimizer, gk, P)

        # Measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        if i % print_freq == 0 or i == len(train_loader)-1:
            print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')
        

def validate(val_loader, model, criterion):
    # Run evaluation
    losses = AverageMeter()
    top1 = AverageMeter()

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.to(device)
            input_var = input.to(device)
            target_var = target.to(device)

            # Compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()

            # Measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            if i % print_freq == 0:
                print(f'Test: [{i}/{len(val_loader)}]\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})')

    print(f' * Prec@1 {top1.avg:.3f}')
    return top1.avg


if __name__ == "__main__":
    # Model Structure
    model = resnet8()
    model.to(device) # cpu is ok

    # Load sampled model parameters
    W = []
    for i in range(params_number):
        model.load_state_dict(torch.load(os.path.join(trajectory_dir, str(i) + ".pt")))
        W.append(get_model_param_vec(model))

    W = np.array(W)
    print("W: ", W.shape)

    # from scratch
    model.load_state_dict(torch.load(os.path.join(trajectory_dir, str(0) + ".pt")))

    # Obtain base variables through PCA
    pca = PCA(n_components=n_components)
    pca.fit(W)
    P = np.array(pca.components_)
    print(pca.explained_variance_)
    print("P: ", P.shape)
    plt.bar([i + 1 for i in range(n_components)], pca.explained_variance_ratio_)
    plt.title("pca ratio")
    plt.ylabel("percent")
    plt.ylim(0, 1)
    plt.savefig("pca_ratio.pdf")

    P = torch.from_numpy(P).float().to(device)

    # Prepare Dataloader
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=128, shuffle=False, num_workers=workers, pin_memory=True)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 50])

    print('Train:', (epochs))
    best_prec1 = 0
    for epoch in range(epochs):
        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, P)
        lr_scheduler.step()

        # Evaluate on validation set
        prec1 = validate(val_loader, model, criterion)
        best_prec1 = max(prec1, best_prec1)

    print('best_prec1:', best_prec1)
