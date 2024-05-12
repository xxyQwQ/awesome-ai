import os
import io
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def __len__(self):
        return len(self.idx2word)

    def append(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        assert os.path.exists(path)
        with io.open(path, 'r', encoding='utf8') as file:
            for line in file:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.append(word)
        indices_list = []
        with io.open(path, 'r', encoding='utf8') as file:
            for line in file:
                words = line.split() + ['<eos>']
                indices = []
                for word in words:
                    indices.append(self.dictionary.word2idx[word])
                indices_list.append(torch.tensor(indices, dtype=torch.int64))
        return torch.cat(indices_list)
