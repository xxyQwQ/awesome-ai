import sed_eval
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from utils import function


def get_audio_tagging_df(df):
    return df.groupby('filename')['event_label'].unique().reset_index()


def audio_tagging_results(reference, estimated, label_to_idx):
    def na_values(val):
        if type(val) is np.ndarray:
            return val
        elif isinstance(val, list):
            return np.array(val)
        if pd.isna(val):
            return np.zeros(len(label_to_idx))
        return val

    if 'event_label' in reference.columns:
        reference = get_audio_tagging_df(reference)
        estimated = get_audio_tagging_df(estimated)
        ref_labels = reference['event_label'].apply(lambda x: function.encode_label(x, label_to_idx))
        reference['event_label'] = ref_labels
        est_labels = estimated['event_label'].apply(lambda x: function.encode_label(x, label_to_idx))
        estimated['event_label'] = est_labels

    matching = reference.merge(estimated, how='outer', on='filename', suffixes=['_ref', '_pred'])
    ret_df = pd.DataFrame(columns=['label', 'f1', 'precision', 'recall'])

    if not estimated.empty:
        matching['event_label_pred'] = matching.event_label_pred.apply(na_values)
        matching['event_label_ref'] = matching.event_label_ref.apply(na_values)
        y_true = np.vstack(matching['event_label_ref'].values)
        y_pred = np.vstack(matching['event_label_pred'].values)
        ret_df.loc[:, 'label'] = label_to_idx.keys()

        for avg in [None, 'macro', 'micro']:
            avg_f1 = f1_score(y_true, y_pred, average=avg)
            avg_pre = precision_score(y_true, y_pred, average=avg)
            avg_rec = recall_score(y_true, y_pred, average=avg)

            if avg == None:
                ret_df.loc[:, 'precision'] = avg_pre
                ret_df.loc[:, 'recall'] = avg_rec
                ret_df.loc[:, 'f1'] = avg_f1
            else:
                ret_df = ret_df._append(
                    {'label': avg, 'precision': avg_pre, 'recall': avg_rec, 'f1': avg_f1},
                    ignore_index=True
                )

    return ret_df


def get_event_list_current_file(df, fname):
    event_file = df[df['filename'] == fname]
    if len(event_file) == 1:
        if pd.isna(event_file['event_label'].iloc[0]):
            event_list_for_current_file = [{'filename': fname}]
        else:
            event_list_for_current_file = event_file.to_dict('records')
    else:
        event_list_for_current_file = event_file.to_dict('records')
    return event_list_for_current_file


def event_based_evaluation_df(reference, estimated, t_collar=0.2, percentage_of_length=0.2):
    evaluated_files = reference['filename'].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    event_based_metric = sed_eval.sound_event.EventBasedMetrics(
        event_label_list=classes,
        t_collar=t_collar,
        percentage_of_length=percentage_of_length,
        empty_system_output_handling='zero_score'
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(reference, fname)
        estimated_event_list_for_current_file = get_event_list_current_file(estimated, fname)
        event_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file,
        )

    return event_based_metric


def segment_based_evaluation_df(reference, estimated, time_resolution=1.):
    evaluated_files = reference['filename'].unique()

    classes = []
    classes.extend(reference.event_label.dropna().unique())
    classes.extend(estimated.event_label.dropna().unique())
    classes = list(set(classes))

    segment_based_metric = sed_eval.sound_event.SegmentBasedMetrics(
        event_label_list=classes,
        time_resolution=time_resolution
    )

    for fname in evaluated_files:
        reference_event_list_for_current_file = get_event_list_current_file(reference, fname)
        estimated_event_list_for_current_file = get_event_list_current_file(estimated, fname)
        segment_based_metric.evaluate(
            reference_event_list=reference_event_list_for_current_file,
            estimated_event_list=estimated_event_list_for_current_file
        )

    return segment_based_metric


def compute_metrics(valid_df, pred_df, time_resolution=1.0):
    metric_event = event_based_evaluation_df(valid_df, pred_df, t_collar=0.200, percentage_of_length=0.2)
    metric_segment = segment_based_evaluation_df(valid_df, pred_df, time_resolution=time_resolution)
    return metric_event, metric_segment
