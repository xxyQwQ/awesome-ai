import os
import sys
import time

import hydra
from omegaconf import OmegaConf
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, AdamW
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from utils.logger import Logger
from utils.function import count_parameter, adjust_alpha, split_batch, detach_hidden
from utils.dataset import Corpus
from model.rnn import RNN
from model.transformer import Transformer
from model.gpt import GPT


def train(config, model, dataset, criterion, optimizer, writer, epoch):
    model.train()
    total_loss = 0
    start_time = time.time()
    if config.model in ['rnn', 'lstm', 'gru']:
        hidden = model.init_hidden(config.batch_size)
    for batch, index in enumerate(range(0, dataset.shape[0] - 1, config.sequence_length)):
        length = min(config.sequence_length, dataset.shape[0] - index - 1)
        source = dataset[index:index + length]
        target = dataset[index + 1:index + length + 1]
        optimizer.zero_grad()
        if config.model in ['rnn', 'lstm', 'gru']:
            hidden = detach_hidden(hidden)
            if config.reuse_hidden:
                output, hidden = model(source, hidden)
            else:
                output, _ = model(source, hidden)
        elif config.model in ['transformer', 'gpt']:
            output = model(source)
        loss = criterion(output.view(-1, config.num_tokens), target.view(-1))
        loss.backward()
        clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        total_loss += loss.item()
        if batch % config.log_interval == 0 and batch > 0:
            axis = (epoch - 1) * dataset.shape[0] // config.sequence_length + batch
            loss = total_loss / config.log_interval
            perplexity = np.exp(loss)
            writer.add_scalar('train/loss', loss, axis)
            writer.add_scalar('train/perplexity', perplexity, axis)
            print('(train) | epoch {:3d} | batch {:5d}/{:5d} | loss {:5.2f} | perplexity {:8.2f} | {:5.2f} ms/batch |'.format(epoch, batch, dataset.shape[0] // config.sequence_length, loss, perplexity, (time.time() - start_time) * 1000 / config.log_interval))
            total_loss = 0
            start_time = time.time()


def evaluate(config, model, dataset, criterion):
    model.eval()
    total_loss = 0
    if config.model in ['rnn', 'lstm', 'gru']:
        hidden = model.init_hidden(config.batch_size)
    with torch.no_grad():
        for index in range(0, dataset.shape[0] - 1, config.sequence_length):
            length = min(config.sequence_length, dataset.shape[0] - index - 1)
            source = dataset[index:index + length]
            target = dataset[index + 1:index + length + 1]
            if config.model in ['rnn', 'lstm', 'gru']:
                hidden = detach_hidden(hidden)
                if config.reuse_hidden:
                    output, hidden = model(source, hidden)
                else:
                    output, _ = model(source, hidden)
            elif config.model in ['transformer', 'gpt']:
                output = model(source)
            loss = criterion(output.view(-1, config.num_tokens), target.view(-1))
            total_loss += source.shape[0] * loss.item()
    return total_loss / dataset.shape[0]


@hydra.main(version_base=None, config_path='./config', config_name='main')
def main(config):
    # load configuration
    checkpoint_path = str(config.checkpoint)
    config.checkpoint = str(checkpoint_path)
    sys.stdout = Logger(os.path.join(checkpoint_path, 'main.log'))
    writer = SummaryWriter(os.path.join(checkpoint_path, 'tensorboard'))
    device = torch.device('cuda') if config.device == 'gpu' else torch.device('cpu')
    config.device = str(device)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # load dataset
    corpus = Corpus(config.dataset)
    config.num_tokens = len(corpus.dictionary)
    dataset_train = split_batch(device, corpus.train, config.batch_size)
    dataset_valid = split_batch(device, corpus.valid, config.batch_size)
    dataset_test = split_batch(device, corpus.test, config.batch_size)

    # create model
    if config.model in ['rnn', 'lstm', 'gru']:
        model = RNN(config.num_tokens, config.embedding_dims, config.hidden_dims, config.num_layers, config.dropout, config.tied, config.model)
    elif config.model == 'transformer':
        model = Transformer(config.num_tokens, config.embedding_dims, config.hidden_dims, config.num_heads, config.num_layers, config.dropout, config.max_length)
    elif config.model == 'gpt':
        model = GPT(config.num_tokens, config.embedding_dims, config.hidden_dims, config.num_heads, config.num_layers, config.dropout, config.max_length)
    model.to(device)
    print('Total number of parameters: {:.2f}M'.format(count_parameter(model) / 1e6))

    # create optimizer
    criterion = nn.NLLLoss()
    if config.model in ['rnn', 'lstm', 'gru']:
        optimizer = Adam(model.parameters(), lr=config.learning_rate)
    elif config.model in ['transformer', 'gpt']:
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # train model
    print(OmegaConf.to_yaml(config))
    best_loss = float('inf')
    learning_rate = config.learning_rate
    for epoch in range(1, config.num_epochs + 1):
        train(config, model, dataset_train, criterion, optimizer, writer, epoch)
        loss = evaluate(config, model, dataset_valid, criterion)
        perplexity = np.exp(loss)
        writer.add_scalar('valid/loss', loss, epoch)
        writer.add_scalar('valid/perplexity', perplexity, epoch)
        print('(valid) | epoch {:3d} | loss {:5.2f} | perplexity {:8.2f} |'.format(epoch, loss, perplexity))
        if loss < best_loss:
            best_loss = loss
            with open(os.path.join(checkpoint_path, 'model.pth'), 'wb') as file:
                torch.save(model, file)
        else:
            learning_rate *= config.decay_rate
            adjust_alpha(optimizer, learning_rate)

    # test model
    with open(os.path.join(checkpoint_path, 'model.pth'), 'rb') as file:
        model = torch.load(file)
    loss = evaluate(config, model, dataset_test, criterion)
    perplexity = np.exp(loss)
    writer.add_scalar('test/loss', loss, 0)
    writer.add_scalar('test/perplexity', perplexity, 0)
    print('(test) | loss {:5.2f} | perplexity {:8.2f} |'.format(loss, perplexity))
    writer.close()


if __name__ == '__main__':
    main()
