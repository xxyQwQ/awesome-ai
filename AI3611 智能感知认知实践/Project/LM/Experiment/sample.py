import os

import hydra
from omegaconf import OmegaConf
import torch

from utils.dataset import Corpus
from utils.function import detach_hidden


@hydra.main(version_base=None, config_path='./config', config_name='sample')
def main(config):
    # load configuration
    checkpoint_path = str(config.checkpoint)
    device = torch.device('cuda') if config.device == 'gpu' else torch.device('cpu')
    config.device = str(device)
    torch.manual_seed(config.seed)

    # load dataset
    corpus = Corpus(config.dataset)
    config.num_tokens = len(corpus.dictionary)

    # load model
    with open(os.path.join(checkpoint_path, 'model.pth'), 'rb') as file:
        model = torch.load(file, map_location=device)
    model.eval()

    # generate text
    print(OmegaConf.to_yaml(config))
    input = torch.randint(config.num_tokens, (1, 1), dtype=torch.long).to(device)
    if config.model in ['rnn', 'lstm', 'gru']:
        hidden = model.init_hidden(1)
    with open(os.path.join(checkpoint_path, 'sample.txt'), 'w') as file:
        with torch.no_grad():
            for word in range(1, config.num_words + 1):
                if config.model in ['rnn', 'lstm', 'gru']:
                    hidden = detach_hidden(hidden)
                    output, hidden = model(input, hidden)
                    distribution = output.squeeze().div(config.temperature).exp().cpu()
                    index = torch.multinomial(distribution, 1).item()
                    input = torch.LongTensor([[index]]).to(device)
                elif config.model in ['transformer', 'gpt']:
                    output = model(input)
                    distribution = output[-1].squeeze().div(config.temperature).exp().cpu()
                    index = torch.multinomial(distribution, 1).item()
                    input = torch.cat([input, torch.LongTensor([[index]]).to(device)], 0)
                file.write(corpus.dictionary.idx2word[index])
                file.write('\n' if word % 20 == 0 else ' ')
                if word % config.log_interval == 0:
                    print('(sample) | word {:5d}/{:5d}'.format(word, config.num_words))


if __name__ == '__main__':
    main()
