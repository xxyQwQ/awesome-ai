import os
import sys
import time
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from model.cnn import MNIST_CNN
from utils.timer import get_time_stamp
from utils.logger import Logger, Recorder, logger_print
from utils.analyzer import visualize_record, summarize_record


def run_cnn(args):
    # create logs
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    time_stamp = get_time_stamp()
    save_path = os.path.join(args.save_path, 'cnn_{}'.format(time_stamp))
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger_path = os.path.join(save_path, 'log.txt')
    sys.stdout = Logger(logger_path)
    logger_print('info', 'create {} for current task\n\tlogs will be saved in {}\n\targuments are listed below\n\t{}\n'.format(save_path, logger_path, args))

    # config device
    if args.device == 'cpu':
        device = 'cpu'
    elif args.device == 'gpu':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    logger_print('info', 'task is expected to run on {}\n\ttask actually runs on {}\n'.format(args.device, device))

    # load dataset
    dataset_train = MNIST('./dataset', train=True, transform=transforms.ToTensor(), download=True)
    dataset_test = MNIST('./dataset', train=False, transform=transforms.ToTensor(), download=True)
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger_print('info', 'dataset is loaded successfully\n\ttraining set contains {} samples\n\ttest set contains {} samples\n'.format(len(dataset_train), len(dataset_test)))

    # config model
    model = MNIST_CNN()
    model.to(device)
    logger_print('info', 'model is loaded successfully\n\tnetwork structure is listed below\n\t{}\n'.format(model))

    # config optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    logger_print('info', 'optimizer is initialized successfully\n\tusing {} optimizer with learning rate {:.4f}\n'.format(args.optimizer, args.learning_rate))
    
    # config loss function and scheduler
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs)
    logger_print('info', 'loss function and scheduler are initialized successfully\n\tusing CrossEntropyLoss and CosineAnnealingLR\n')

    # start training
    recorder = Recorder('epoch', 'loss_train', 'accuracy_train', 'time_train', 'loss_test', 'accuracy_test', 'time_test')
    logger_print('info', 'training process will last for {} epochs\n'.format(args.num_epochs))
    
    # main loop
    for epoch in range(args.num_epochs):
        logger_print('info', 'epoch {} is started'.format(epoch))

        # train step
        loss_train = 0
        correct_train = 0
        start_train = time.time()
        model.train()

        for images, labels in loader_train:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_train += labels.numel() * loss.item()
            correct_train += (outputs.argmax(dim=1) == labels).sum().item()
        
        scheduler.step()
        loss_train /= len(dataset_train)
        accuracy_train = correct_train / len(dataset_train)
        time_train = time.time() - start_train
        
        # test step
        loss_test = 0
        correct_test = 0
        start_test = time.time()
        model.eval()
        
        with torch.no_grad():
            for images, labels in loader_test:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss_test += labels.numel() * loss.item()
                correct_test += (outputs.argmax(dim=1) == labels).sum().item()
        
        loss_test /= len(dataset_test)
        accuracy_test = correct_test / len(dataset_test)
        time_test = time.time() - start_test

        # report results
        recorder.record({'epoch': epoch, 'loss_train': loss_train, 'accuracy_train': accuracy_train, 'time_train': time_train, 'loss_test': loss_test, 'accuracy_test': accuracy_test, 'time_test': time_test})
        logger_print('info', 'epoch {} is finished\n\ttraining step takes {:.2f} seconds with loss {:.4f} and accuracy {:.2%}\n\ttest step takes {:.2f} seconds with loss {:.4f} and accuracy {:.2%}\n'.format(epoch, time_train, loss_train, accuracy_train, time_test, loss_test, accuracy_test))

    # analyze results
    visualize_record(save_path, recorder.records)
    summarize_record(save_path, recorder.records)
    logger_print('info', 'records are analyzed successfully\n\tfigures and summary are saved in {}\n'.format(save_path))

    # save results
    recorder_path = os.path.join(save_path, 'record.pkl')
    recorder.save(recorder_path)
    model_path = os.path.join(save_path, 'model.pth')
    torch.save(model, model_path)
    logger_print('info', 'training process is finished\n\trecords are saved in {}\n\tmodel is saved in {}\n'.format(recorder_path, model_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--device', default='gpu', type=str, choices=['cpu', 'gpu'], help='device')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('-j', '--num_workers', default=0, type=int, help='number of workers')
    parser.add_argument('-o', '--optimizer', default='adam', type=str, choices=['sgd', 'adam'], help='optimizer')
    parser.add_argument('-r', '--learning_rate', default=1e-3, type=float, help='learning rate')
    parser.add_argument('-e', '--num_epochs', default=100, type=int, help='number of epochs')
    parser.add_argument('-s', '--save_path', default='./checkpoint', type=str, help='save path')
    args = parser.parse_args()
    run_cnn(args)
