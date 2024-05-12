import sys
import yaml
from pprint import pformat

import numpy as np
from torch import nn
from scipy import ndimage
from loguru import logger
from sklearn import preprocessing


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, output):
        return nn.functional.binary_cross_entropy(output['clip_prob'], output['target'])


def parse_config_or_kwargs(config_file, **kwargs):
    with open(config_file) as reader:
        yaml_config = yaml.load(reader, Loader=yaml.FullLoader)
    arguments = dict(yaml_config, **kwargs)
    return arguments


def dump_config(config_file, config):
    with open(config_file, 'w') as writer:
        yaml.dump(config, writer, default_flow_style=False)


def encode_label(label, label_to_idx):
    target = np.zeros(len(label_to_idx))
    if isinstance(label, str):
        label = label.split(',')
    for lb in label:
        target[label_to_idx[lb]] = 1
    return target


def find_contiguous_regions(activity_array):
    change_indices = np.logical_xor(activity_array[1:], activity_array[:-1]).nonzero()[0]
    change_indices += 1
    if activity_array[0]:
        change_indices = np.r_[0, change_indices]
    if activity_array[-1]:
        change_indices = np.r_[change_indices, activity_array.size]
    return change_indices.reshape((-1, 2))


def split_train_cv(data_frame, frac=0.9, y=None, stratified=True):
    if stratified:
        from skmultilearn.model_selection import iterative_train_test_split
        index_train, _, index_cv, _ = iterative_train_test_split(data_frame.index.values.reshape(-1, 1), y, 1 - frac)
        train_data = data_frame[data_frame.index.isin(index_train.squeeze())]
        cv_data = data_frame[data_frame.index.isin(index_cv.squeeze())]
    else:
        train_data = data_frame.sample(frac=frac, random_state=10)
        cv_data = data_frame[~data_frame.index.isin(train_data.index)]
    return train_data, cv_data


def pprint_dict(in_dict, outputfun=sys.stdout.write, formatter='yaml'):
    if formatter == 'yaml':
        format_fun = yaml.dump
    elif formatter == 'pretty':
        format_fun = pformat
    for line in format_fun(in_dict).split('\n'):
        outputfun(line)


def create_logger(config_file):
    log_format = '[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}'
    logger.configure(handlers=[{'sink': sys.stderr, 'format': log_format}])
    logger.add(config_file, enqueue=True, format=log_format)
    return logger


def binarize(pred, threshold=0.5):
    if pred.ndim == 3:
        return np.array([preprocessing.binarize(sub, threshold=threshold) for sub in pred])
    else:
        return preprocessing.binarize(pred, threshold=threshold)


def median_filter(x, window_size, threshold=0.5):
    x = binarize(x, threshold=threshold)
    if x.ndim == 3:
        size = (1, window_size, 1)
    elif x.ndim == 2 and x.shape[0] == 1:
        size = (1, window_size)
    elif x.ndim == 2 and x.shape[0] > 1:
        size = (window_size, 1)
    return ndimage.median_filter(x, size=size)


def _decode_with_timestamps(idx_to_label, labels):
    result_labels = []
    for i, label_column in enumerate(labels.T):
        change_indices = find_contiguous_regions(label_column)
        for row in change_indices:
            result_labels.append((idx_to_label[i], row[0], row[1]))
    return result_labels


def decode_with_timestamps(idx_to_label, labels):
    if labels.ndim == 3:
        return [_decode_with_timestamps(idx_to_label, lab) for lab in labels]
    else:
        return _decode_with_timestamps(idx_to_label, labels)


def predictions_to_time(df, ratio):
    df.onset = df.onset * ratio
    df.offset = df.offset * ratio
    return df
