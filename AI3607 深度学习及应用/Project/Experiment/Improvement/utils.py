import os
import numpy as np
import matplotlib.pyplot as plt
import jittor as jt
from jittor import nn


def train(model, loader, total_epoch=10, learning_rate=1e-3, loss_function=None, optimizer=None, scheduler=None, save_path='./model', save_name='model'):
    if loss_function is None:
        loss_function = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = nn.Adam(model.parameters(), lr=learning_rate)
    if scheduler is not None:
        step_size, decay_rate = scheduler
        scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=decay_rate)
    
    loader_train, loader_test = loader
    record_train, record_test = [], []

    for epoch in range(1, total_epoch+1):
        print('[progress] epoch: {}'.format(epoch))

        model.train()
        epoch_bias, fragment_count, puzzle_count = 0, 0, 0
        for images, labels in loader_train:
            outputs = model(images)
            loss = loss_function(outputs.view(-1, model.block), labels.view(-1))
            optimizer.step(loss)
            correct = (outputs.argmax(dim=2)[0] == labels).sum(dim=1)
            epoch_bias += loss.data[0]
            fragment_count += correct.sum().data[0]
            puzzle_count += (correct == model.block).sum().data[0]
        fragment_accuracy, puzzle_accuracy = fragment_count / model.block / len(loader_train), puzzle_count / len(loader_train)
        record_train.append([epoch_bias, fragment_accuracy, puzzle_accuracy])
        print('(train) epoch loss: {:.4f}, fragment accuracy: {:.2%}, puzzle accuracy: {:.2%}'.format(epoch_bias, fragment_accuracy, puzzle_accuracy))

        model.eval()
        epoch_bias, fragment_count, puzzle_count = 0, 0, 0
        with jt.no_grad():
            for images, labels in loader_test:
                outputs = model(images)
                loss = loss_function(outputs.view(-1, model.block), labels.view(-1))
                correct = (outputs.argmax(dim=2)[0] == labels).sum(dim=1)
                epoch_bias += loss.data[0]
                fragment_count += correct.sum().data[0]
                puzzle_count += (correct == model.block).sum().data[0]
        fragment_accuracy, puzzle_accuracy = fragment_count / model.block / len(loader_test), puzzle_count / len(loader_test)
        record_test.append([epoch_bias, fragment_accuracy, puzzle_accuracy])
        print('(test) epoch loss: {:.4f}, fragment accuracy: {:.2%}, puzzle accuracy: {:.2%}'.format(epoch_bias, fragment_accuracy, puzzle_accuracy))

        if scheduler is not None:
            scheduler.step()

    model.save(os.path.join(save_path, '{}_epoch_{}.pkl'.format(save_name, total_epoch)))
    return np.array(record_train), np.array(record_test)


def plot(record_train, record_test, save_path='./figure', save_name='figure'):
    plt.figure(figsize=(8, 6))
    plt.plot(record_train[:, 0], label='train')
    plt.plot(record_test[:, 0], label='test')
    plt.xlabel('epoch')
    plt.ylabel('epoch loss')
    plt.title('Epoch Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, '{}_epoch_loss.png'.format(save_name)))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(record_train[:, 1], label='train')
    plt.plot(record_test[:, 1], label='test')
    plt.xlabel('epoch')
    plt.ylabel('fragment accuracy')
    plt.title('Fragment Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, '{}_fragment_accuracy.png'.format(save_name)))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(record_train[:, 2], label='train')
    plt.plot(record_test[:, 2], label='test')
    plt.xlabel('epoch')
    plt.ylabel('puzzle accuracy')
    plt.title('Puzzle Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, '{}_puzzle_accuracy.png'.format(save_name)))
    plt.close()
