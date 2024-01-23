import os
import numpy as np
import matplotlib.pyplot as plt


def visualize_record(path, record):
    plt.figure(figsize=(8, 6))
    plt.plot(record['epoch'], record['loss_train'], label='train')
    plt.plot(record['epoch'], record['loss_test'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('loss curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path, 'loss.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(record['epoch'], record['accuracy_train'], label='train')
    plt.plot(record['epoch'], record['accuracy_test'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('accuracy curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path, 'accuracy.png'))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(record['epoch'], record['time_train'], label='train')
    plt.plot(record['epoch'], record['time_test'], label='test')
    plt.xlabel('epoch')
    plt.ylabel('time')
    plt.title('time curve')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(path, 'time.png'))
    plt.close()


def summarize_record(path, record):
    with open(os.path.join(path, 'summary.txt'), 'w') as file:
        best_epoch = np.argmax(record['accuracy_test'])
        file.write('peak accuracy is achieved at epoch {}\n'.format(best_epoch))
        file.write('corresponding training accuracy is {:.2%}\n'.format(record['accuracy_train'][best_epoch]))
        file.write('corresponding test accuracy is {:.2%}\n'.format(record['accuracy_test'][best_epoch]))

        time_total = np.sum(record['time_train']) + np.sum(record['time_test'])
        file.write('finish {} epochs in {:.2f} seconds\n'.format(len(record['epoch']), time_total))
        file.write('in average {:.2f} seconds per epoch\n'.format(time_total / len(record['epoch'])))
        file.write('where {:.2f} seconds for training and {:.2f} seconds for test\n'.format(np.mean(record['time_train']), np.mean(record['time_test'])))
