import numpy as np
import torch

from utils.vocab import PAD, UNK


class Word2vecUtils():

    def __init__(self, word2vec_file):
        super(Word2vecUtils, self).__init__()
        self.word2vec = {}
        self.read_from_file(word2vec_file)

    def load_embeddings(self, module, vocab, device='cpu'):
        """ Initialize the embedding with glove and char embedding
        """
        emb_size = module.weight.data.size(-1)
        outliers = 0
        for word in vocab.word2id:
            if word == PAD: # PAD symbol is always 0-vector
                module.weight.data[vocab[PAD]] = torch.zeros(emb_size, dtype=torch.float, device=device)
                continue
            word_emb = self.word2vec.get(word, self.word2vec[UNK])
            module.weight.data[vocab[word]] = torch.tensor(word_emb, dtype=torch.float, device=device)
        return 1 - outliers / float(len(vocab))

    def read_from_file(self, word2vec_file):
        with open(word2vec_file, 'r') as f:
            for line in f:
                line = line.strip('\n')
                items = line.split(' ')
                if len(items) <= 2:
                    continue
                word = items[0]
                vector = np.fromstring(' '.join(items[1:]), dtype=float, sep=' ')
                self.word2vec[word] = vector
