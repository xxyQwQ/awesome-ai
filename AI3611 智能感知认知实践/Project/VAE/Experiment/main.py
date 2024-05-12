import os
import sys
import time

import hydra
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.optim import Adam

from utils.logger import Logger, Recorder
from utils.function import create_dataset, create_model, compute_loss, plot_curve, plot_sample


@hydra.main(version_base=None, config_path='./config', config_name='main')
def main(config):
    # load configuration
    checkpoint_path = str(config.checkpoint)
    dataset_path = str(config.dataset)
    device = torch.device('cuda') if config.device == 'gpu' else torch.device('cpu')
    parameter = config.parameter

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint_path, 'main.log'))
    config.checkpoint = str(checkpoint_path)
    config.device = str(device)
    print(OmegaConf.to_yaml(config))

    # create model
    model = create_model(device, parameter.input_dims, parameter.latent_dims, parameter.hidden_dims, parameter.hidden_layers)

    # create optimizer
    optimizer = Adam(model.parameters(), lr=parameter.learning_rate)

    # load dataset
    dataset_train, dataset_test, loader_train, loader_test = create_dataset(dataset_path, parameter.batch_size, parameter.num_workers)
    print('train dataset: {} samples, test dataset: {} samples\n'.format(len(dataset_train), len(dataset_test)))

    # start training
    recorder = Recorder('epoch', 'train_loss', 'train_loss_reconstruction', 'train_loss_kl_divergence', 'train_time', 'test_loss', 'test_loss_reconstruction', 'test_loss_kl_divergence', 'test_time')
    print('total epochs: {}\n'.format(parameter.num_epochs))

    for epoch in range(1, parameter.num_epochs + 1):
        print('epoch start: {} / {}'.format(epoch, parameter.num_epochs))

        # train
        train_loss, train_loss_reconstruction, train_loss_kl_divergence = 0, 0, 0
        train_start = time.time()
        model.train()
        for input, _ in tqdm(loader_train):
            input = input.to(device)
            output, mu, logvar, _ = model(input)
            optimizer.zero_grad()
            loss_reconstruction, loss_kl_divergence = compute_loss(input, output, mu, logvar)
            loss = loss_reconstruction + parameter.beta_value * loss_kl_divergence
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_loss_reconstruction += loss_reconstruction.item()
            train_loss_kl_divergence += loss_kl_divergence.item()
        train_loss /= len(dataset_train)
        train_loss_reconstruction /= len(dataset_train)
        train_loss_kl_divergence /= len(dataset_train)
        train_time = time.time() - train_start

        # test
        test_loss, test_loss_reconstruction, test_loss_kl_divergence = 0, 0, 0
        test_start = time.time()
        model.eval()
        with torch.no_grad():
            for input, _ in tqdm(loader_test):
                input = input.to(device)
                output, mu, logvar, _ = model(input)
                loss_reconstruction, loss_kl_divergence = compute_loss(input, output, mu, logvar)
                loss = loss_reconstruction + parameter.beta_value * loss_kl_divergence
                test_loss += loss.item()
                test_loss_reconstruction += loss_reconstruction.item()
                test_loss_kl_divergence += loss_kl_divergence.item()
        test_loss /= len(dataset_test)
        test_loss_reconstruction /= len(dataset_test)
        test_loss_kl_divergence /= len(dataset_test)
        test_time = time.time() - test_start

        # record
        recorder.record({'epoch': epoch, 'train_loss': train_loss, 'train_loss_reconstruction': train_loss_reconstruction, 'train_loss_kl_divergence': train_loss_kl_divergence, 'train_time': train_time, 'test_loss': test_loss, 'test_loss_reconstruction': test_loss_reconstruction, 'test_loss_kl_divergence': test_loss_kl_divergence, 'test_time': test_time})
        print('epoch finish: {} / {}'.format(epoch, parameter.num_epochs))
        print('(train) loss: {:.6f}, reconstruction: {:.6f}, kl_divergence: {:.6f}, time: {:.2f}'.format(train_loss, train_loss_reconstruction, train_loss_kl_divergence, train_time))
        print('(test) loss: {:.6f}, reconstruction: {:.6f}, kl_divergence: {:.6f}, time: {:.2f}\n'.format(test_loss, test_loss_reconstruction, test_loss_kl_divergence, test_time))

        # save
        if epoch % parameter.save_interval == 0:
            save_path = os.path.join(checkpoint_path, 'epoch_{}'.format(epoch))
            os.makedirs(save_path, exist_ok=True)
            recorder.save(os.path.join(save_path, 'record.pkl'))
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
            plot_curve(recorder.records, os.path.join(save_path, 'loss.png'))
            plot_sample(device, model, dataset_test, os.path.join(save_path, 'sample.png'))
            print('save checkpoint: {}\n'.format(save_path))


if __name__ == '__main__':
    main()
