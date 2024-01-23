import os
import sys
import time

import hydra
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from spikingjelly.activation_based import functional

from utils.logger import Logger, Recorder
from utils.dataset import create_dataset
from utils.function import inherit_model, create_scheduler, reshape_image, visualize_record, summarize_record


@hydra.main(version_base=None, config_path='./config', config_name='finetune')
def main(config):
    # load configuration
    checkpoint = str(config.checkpoint)
    device = torch.device('cuda') if str(config.device) == 'gpu' else torch.device('cpu')
    experiment = config.experiment

    # create logger
    sys.stdout = Logger(os.path.join(checkpoint, 'finetune.log'))
    config.checkpoint = str(checkpoint)
    config.device = str(device)
    print(OmegaConf.to_yaml(config))

    # create model
    model = inherit_model(device, experiment)
    functional.set_step_mode(model, 'm')

    # create optimizer
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=experiment.learning_rate, weight_decay=experiment.weight_decay)
    scheduler = create_scheduler(optimizer, experiment)

    # load dataset
    dataset_train, dataset_test, loader_train, loader_test = create_dataset(experiment)
    print('train dataset: {} samples, test dataset: {} samples\n'.format(len(dataset_train), len(dataset_test)))

    # start training
    recorder = Recorder('epoch', 'loss_train', 'accuracy_train', 'time_train', 'loss_test', 'accuracy_test', 'time_test')
    print('num_epochs: {}\n'.format(experiment.num_epochs))

    for epoch in range(1, experiment.num_epochs + 1):
        print('epoch start: {} / {}'.format(epoch, experiment.num_epochs))

        # train
        loss_train = 0
        correct_train = 0
        start_train = time.time()
        model.train()

        for image, label in tqdm(loader_train):
            image, label = image.to(device), label.to(device)
            image = reshape_image(image, experiment.time_steps)

            optimizer.zero_grad()
            prediction = model(image)
            loss = criterion(prediction, label)
            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            correct_train += (prediction.argmax(dim=1) == label).sum().item()
            functional.reset_net(model)

        scheduler.step()
        loss_train /= len(dataset_train)
        accuracy_train = correct_train / len(dataset_train)
        time_train = time.time() - start_train

        # test
        loss_test = 0
        correct_test = 0
        start_test = time.time()
        model.eval()

        with torch.no_grad():
            for image, label in tqdm(loader_test):
                image, label = image.to(device), label.to(device)
                image = reshape_image(image, experiment.time_steps)

                prediction = model(image)
                loss = criterion(prediction, label)

                loss_test += loss.item()
                correct_test += (prediction.argmax(dim=1) == label).sum().item()
                functional.reset_net(model)

        loss_test /= len(dataset_test)
        accuracy_test = correct_test / len(dataset_test)
        time_test = time.time() - start_test

        # report
        recorder.record({'epoch': epoch, 'loss_train': loss_train, 'accuracy_train': accuracy_train, 'time_train': time_train, 'loss_test': loss_test, 'accuracy_test': accuracy_test, 'time_test': time_test})
        print('epoch finish: {} / {}'.format(epoch, experiment.num_epochs))
        print('loss_train: {:.6f}, accuracy_train: {:.2%}, time_train: {:.2f}'.format(loss_train, accuracy_train, time_train))
        print('loss_test: {:.6f}, accuracy_test: {:.2%}, time_test: {:.2f}\n'.format(loss_test, accuracy_test, time_test))

        # save
        if epoch % experiment.save_interval == 0:
            save_path = os.path.join(checkpoint, 'epoch_{}'.format(epoch))
            os.makedirs(save_path, exist_ok=True)
            visualize_record(save_path, recorder.records)
            summarize_record(save_path, recorder.records)
            recorder.save(os.path.join(save_path, 'record.pkl'))
            torch.save(model.state_dict(), os.path.join(save_path, 'model.pth'))
            print('save checkpoint: {}\n'.format(save_path))


if __name__ == '__main__':
    main()
