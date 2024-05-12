import os
import sys
import itertools

import numpy as np
from loguru import logger

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def get_logger(config_file):
    log_format = '[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}'
    logger.configure(handlers=[{'sink': sys.stderr, 'format': log_format}])
    logger.add(config_file, enqueue=True, format=log_format)
    return logger


def ptb_tokenize(key_to_captions):
    captions_for_image = {}
    for key, caps in key_to_captions.items():
        captions_for_image[key] = []
        for cap in caps:
            captions_for_image[key].append({'caption': cap})
    tokenizer = PTBTokenizer()
    key_to_captions = tokenizer.tokenize(captions_for_image)
    return key_to_captions


def split_data(tgt, img):
    lst = []
    for i in img:
        if os.path.basename(i) in tgt:
            lst.append(i)
    return lst


def words_from_tensors_fn(idx2word, endseq='<end>'):
    def words_from_tensors(captions):
        captoks = []
        for capidx in captions:
            captoks.append(list(itertools.takewhile(
                lambda word: word != endseq,
                map(lambda idx: idx2word[idx], iter(capidx))
            ))[1:])
        return captoks
    return words_from_tensors


def sched_sampling_eps_fn(mode=None):
    def lin_sched_sampling_eps(i, eps=0.1, k=1.0, c=0.05):
        return max(eps, k - c * i)
    def exp_sched_sampling_eps(i, k=0.95):
        return k ** i
    def sig_sched_sampling_eps(i, k=5.0):
        return k / (k + np.exp(i / k))
    if mode is None:
        sched_sampling_eps = lambda _: 1.0
    elif mode == 'linear':
        sched_sampling_eps = lin_sched_sampling_eps
    elif mode == 'exponential':
        sched_sampling_eps = exp_sched_sampling_eps
    elif mode == 'sigmoid':
        sched_sampling_eps = sig_sched_sampling_eps
    return sched_sampling_eps
