import sys
import argparse


args_choices = {
    'model': ['slu_tagging', 'slu_bert', 'slu_bert_rnn', 'slu_transformer', 'slu_rnn_crf', 'slu_bert_crf', 'slu_bert_rnn_crf'],
    'optimizer': ['Adam', 'AdamW'],
    'encoder_cell': ['RNN', 'LSTM', 'GRU'],
}


def add_argument_base(arg_parser):
    # general
    arg_parser.add_argument('--model', type=str, default='slu_tagging', choices=args_choices['model'], help='model name')
    arg_parser.add_argument('--ckpt', type=str, default='./ckpt', help='path to save checkpoint')
    arg_parser.add_argument('--dataroot', type=str, default='./data', help='root of data')
    arg_parser.add_argument('--seed', type=int, default=999, help='Random seed')
    arg_parser.add_argument('--device', type=int, default=-1, help='Use which device: -1 -> cpu ; the index of gpu o.w.')
    arg_parser.add_argument('--testing', action='store_true', help='training or evaluation mode')
    arg_parser.add_argument('--correction', action='store_true', help='whether to use correction')
    arg_parser.add_argument('--refinement', action='store_true', help='whether to use refinement')
    # training
    arg_parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    arg_parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    arg_parser.add_argument('--max_epoch', type=int, default=100, help='terminate after maximum epochs')
    arg_parser.add_argument('--optimizer', type=str, default='Adam', choices=args_choices['optimizer'], help='optimizer')
    arg_parser.add_argument('--num_heads', type=int, default=8, help='number of attention heads')
    # model
    arg_parser.add_argument('--bert_version', type=str, default='hfl/chinese-bert-wwm-ext', help='path of pretrained bert')
    arg_parser.add_argument('--warmup_epoch', type=int, default=3, help='encoder type')
    arg_parser.add_argument('--encoder_cell', type=str, default=None, choices=args_choices['encoder_cell'], help='type of rnn cell')
    arg_parser.add_argument('--dropout', type=float, default=0.2, help='feature dropout rate')
    arg_parser.add_argument('--embed_size', type=int, default=768, help='Size of word embeddings')
    arg_parser.add_argument('--hidden_size', type=int, default=512, help='hidden size')
    arg_parser.add_argument('--num_layer', type=int, default=2, help='number of layer')
    return arg_parser


def init_args(params=sys.argv[1:]):
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_argument_base(arg_parser)
    opt = arg_parser.parse_args(params)
    return opt
