import argparse
import pandas as pd
from tabulate import tabulate

from utils import metrics


def evaluate(args):
    label_df = pd.read_csv(args.label, sep='\t')
    pred_df = pd.read_csv(args.prediction, sep='\t')
    event_result, segment_result = metrics.compute_metrics(
        label_df,
        pred_df,
        time_resolution=1.0
    )
    event_based_results = pd.DataFrame(
        event_result.results_class_wise_average_metrics()['f_measure'],
        index=['event_based']
    )
    segment_based_results = pd.DataFrame(
        segment_result.results_class_wise_average_metrics()['f_measure'],
        index=['segment_based']
    )
    result_quick_report = pd.concat((
        event_based_results,
        segment_based_results,
    ))
    with open(args.target, 'w') as writer:
        print(str(event_result), file=writer)
        print(str(segment_result), file=writer)
        print('Quick report: ', file=writer)
        print(tabulate(result_quick_report, headers='keys', tablefmt='github'), file=writer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction', type=str, required=True)
    parser.add_argument('--label', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)
    args = parser.parse_args()
    evaluate(args)
