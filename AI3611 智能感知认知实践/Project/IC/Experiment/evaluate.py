import os
import argparse
import pandas as pd
from pprint import pprint

from utils.function import ptb_tokenize


def evaluate(args):
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice

    def print_result(metric_name, average_score, score_list):
        best_value = max(score_list)
        best_image = imgnames[score_list.index(best_value)]
        worst_value = min(score_list)
        worst_image = imgnames[score_list.index(worst_value)]
        result = {
            'metric': metric_name,
            'score': f'{average_score:.3f}',
            'best': {
                'name': best_image,
                'value': f'{best_value:.3f}',
                'reference': key_to_refs[best_image],
                'prediction': key_to_pred[best_image]
            },
            'worst': {
                'name': worst_image,
                'value': f'{worst_value:.3f}',
                'reference': key_to_refs[worst_image],
                'prediction': key_to_pred[worst_image]
            }
        }
        pprint(result, stream=writer, width=120, sort_dicts=False)
        print(file=writer)

    prediction_df = pd.read_json(os.path.join(args.checkpoint, 'prediction.json'))
    key_to_pred = dict(zip(prediction_df['img_id'], prediction_df['prediction']))
    imgnames = list(key_to_pred.keys())
    captions = open(args.caption, 'r').read().strip().split('\n')
    key_to_refs = {}
    for row in captions:
        row = row.split('\t')
        row[0] = row[0][:-2]
        if row[0] not in key_to_pred:
            continue
        if row[0] in key_to_refs:
            key_to_refs[row[0]].append(row[1])
        else:
            key_to_refs[row[0]] = [row[1]]
    ptb_key_to_refs = ptb_tokenize(key_to_refs)
    ptb_key_to_pred = ptb_tokenize(key_to_pred)

    scorers = [Bleu(n=4), Rouge(), Meteor(), Cider(), Spice()]
    output = {'SPIDEr': 0}
    with open(os.path.join(args.checkpoint, 'result.txt'), 'w') as writer:
        for scorer in scorers:
            score, scores = scorer.compute_score(ptb_key_to_refs, ptb_key_to_pred)
            method = scorer.method()
            output[method] = score
            if method == 'Bleu':
                for n in range(4):
                    print_result(f'Blue-{n + 1}', score[n], scores[n])
            elif method == 'CIDEr':
                cider_score, cider_scores = score, list(scores)
                print_result(method, score, list(scores))
            elif method == 'SPICE':
                scores = [s['All']['f'] for s in scores]
                spice_score, spice_scores = score, scores
                print_result(method, score, scores)
            else:
                print_result(method, score, list(scores))
        spider_score = (cider_score + spice_score) / 2
        spider_scores = [(x + y) / 2 for x, y in zip(cider_scores, spice_scores)]
        print_result('SPIDEr', spider_score, spider_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--caption', default='./dataset/flickr8k/caption.txt')
    args = parser.parse_args()
    evaluate(args)
