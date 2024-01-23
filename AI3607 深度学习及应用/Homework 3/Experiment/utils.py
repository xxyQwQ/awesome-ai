import os
import numpy as np
import matplotlib.pyplot as plt
import jittor as jt
from jittor import nn


def train(model, loader, total_epoch=10, learning_rate=1e-3, criterion=None, optimizer=None, scheduler=None, save_path='./model', save_name='model', generalize=False):
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = nn.Adam(model.parameters(), lr=learning_rate)
    if scheduler is None:
        scheduler = jt.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
    
    loader_train, loader_test = loader
    record_train, record_test = [], []

    for epoch in range(1, total_epoch+1):
        print('[progress] epoch: {}'.format(epoch))

        model.train()
        bias, correct, counter = 0, [0 for _ in range(10)], [0 for _ in range(10)]
        for images, labels in loader_train:
            outputs = model(images)
            loss = criterion(outputs, labels)
            if generalize:
                omega = jt.Var([1.0])
                loss_ = criterion(omega * outputs, labels)
                gradient = jt.grad(loss_, omega, retain_graph=True)
                loss += jt.sum(jt.sqr(gradient))[0]
            optimizer.step(loss)
            bias += loss.data[0]
            score = outputs.argmax(dim=1)[0] == labels
            for value in range(10):
                index = np.where(labels == value)[0]
                correct[value] += score[index].sum().data[0]
                counter[value] += index.shape[0]
        record_train.append([bias, sum(correct)/sum(counter)])
        print('(train) loss: {:.4f}, accuracy: {:.2%}'.format(bias, sum(correct)/sum(counter)))
        for value in range(10):
            print('\tclass: {}, rate: {:.2%}'.format(value, correct[value]/counter[value]))

        model.eval()
        bias, correct, counter = 0, [0 for _ in range(10)], [0 for _ in range(10)]
        with jt.no_grad():
            for images, labels in loader_test:
                outputs = model(images)
                loss = criterion(outputs, labels)
                bias += loss.data[0]
                score = outputs.argmax(dim=1)[0] == labels
                for value in range(10):
                    index = np.where(labels == value)[0]
                    correct[value] += score[index].sum().data[0]
                    counter[value] += index.shape[0]
        record_test.append([bias, sum(correct)/sum(counter)])
        print('(test) loss: {:.4f}, accuracy: {:.2%}'.format(bias, sum(correct)/sum(counter)))
        for value in range(10):
            print('\tclass: {}, rate: {:.2%}'.format(value, correct[value]/counter[value]))

        scheduler.step()
        if epoch % 10 == 0:
            model.save(os.path.join(save_path, '{}_epoch_{}.pkl'.format(save_name, epoch)))

    return np.array(record_train), np.array(record_test)


def plot(record_train, record_test, save_path='./figure', save_name='figure'):
    plt.figure(figsize=(8, 6))
    plt.plot(record_train[:, 0], label='train')
    plt.plot(record_test[:, 0], label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, '{}_loss.png'.format(save_name)))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(record_train[:, 1], label='train')
    plt.plot(record_test[:, 1], label='test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, '{}_accuracy.png'.format(save_name)))
    plt.close()
