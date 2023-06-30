import time
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


def parse_vad_label(line, frame_size=0.025, frame_shift=0.010):
    """Parse VAD information in each line, and convert it to frame-wise VAD label.

    Args:
        line: e.g. "0.2,3.11 3.48,10.51 10.52,11.02"
        frame_size: frame size (in seconds)
        frame_shift: frame shift (in seconds)
    Returns:
        frames: frame-wise VAD label

    Examples:
        >> label = parse_vad_label("0.3,0.5 0.7,0.9")
        [0, ..., 0, 1, ..., 1, 0, ..., 0, 1, ..., 1]
        >> print(len(label))
        110

    NOTE: The output label length may vary according to the last timestamp in `line`,
    which may not correspond to the real duration of that sample.

    For example, if an audio sample contains 1-sec silence at the end, the resulting
    VAD label will be approximately 1-sec shorter than the sample duration.

    Thus, you need to pad zeros manually to the end of each label to match the number
    of frames in the feature as follows:
        >> feature = extract_feature(audio)    # frames: 320
        >> frames = feature.shape[1]           # here assumes the frame dimension is 1
        >> label = parse_vad_label(vad_line)   # length: 210
        >> import numpy as np
        >> label_pad = np.pad(label, (0, np.maximum(frames - len(label), 0)))[:frames]

    """
    def frame2time(n):
        return n * frame_shift + frame_size / 2
    frames = []
    frame_n = 0
    for time_pairs in line.split():
        start, end = map(float, time_pairs.split(","))
        assert end > start, (start, end)
        while frame2time(frame_n) < start:
            frames.append(0)
            frame_n += 1
        while frame2time(frame_n) <= end:
            frames.append(1)
            frame_n += 1
    return frames


def prediction_to_vad_label(prediction, frame_size=0.025, frame_shift=0.010, threshold=0.5):
    """Convert model prediction to VAD labels.

    Args:
        prediction: predicted speech activity of each frame in one sample
        frame_size: frame size (in seconds)
        frame_shift: frame shift (in seconds)
        threshold: prediction threshold
    Returns:
        vad_label: e.g. "0.31,2.56 2.6,3.89 4.62,7.99 8.85,11.06"

    NOTE: Each frame is converted to the timestamp according to its center time point.
    Thus, the converted labels may not exactly coincide with the original VAD label, depending
    on the specified `frame_size` and `frame_shift`.
    See the following example for more detailed explanation.

    Examples:
        >> label = parse_vad_label("0.31,0.52 0.75,0.92")
        >> prediction_to_vad_label(label)
        '0.31,0.53 0.75,0.92'

    """
    def frame2time(n):
        return n * frame_shift + frame_size / 2
    speech_frames = []
    prev_state = False
    start, end = 0, 0
    end_prediction = len(prediction) - 1
    for i, pred in enumerate(prediction):
        state = pred > threshold
        if not prev_state and state:
            # 0 -> 1
            start = i
        elif not state and prev_state:
            # 1 -> 0
            end = i
            speech_frames.append('{:.2f},{:.2f}'.format(frame2time(start), frame2time(end)))
        elif i == end_prediction and state:
            # 1 -> 1 (end)
            end = i
            speech_frames.append('{:.2f},{:.2f}'.format(frame2time(start), frame2time(end)))
        prev_state = state
    return ' '.join(speech_frames)


def read_label_from_file(path='./data/dev_label.txt', frame_size=0.025, frame_shift=0.010):
    """Read VAD information of all samples, and convert into frame-wise labels (not padded yet).

    Args:
        path: path to the VAD label file.
        frame_size: frame size (in seconds)
        frame_shift: frame shift (in seconds)
    Returns:
        data: dictionary storing the frame-wise VAD information of each sample
            e.g. {"1031-133220-0062": [0, 0, 0, 0, ... ], "1031-133220-0091": [0, 0, 0, 0, ... ]}

    """
    data = {}
    with Path(path).open('r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            sps = line.strip().split(maxsplit=1)
            if len(sps) == 1:
                print(f'Error happened with path="{path}", id="{sps[0]}", value=""')
            else:
                k, v = sps
            if k in data:
                raise RuntimeError(f'{k} is duplicated ({path}:{line_num})')
            data[k] = parse_vad_label(v, frame_size=frame_size, frame_shift=frame_shift)
    return data


def compute_acc(prediction, label):
    """Calculate accuracy of a classification task.

    Args:
        prediction: sequence of probabilities
        label: sequence of class labels
    Returns:
        acc: classification accuracy

    """
    assert len(prediction) == len(label), (len(prediction), len(label))
    return metrics.accuracy_score(prediction, label)


def compute_eer(target_scores, other_scores):
    """Calculate equal error rate.

    Args:
        target_scores: sequence of scores where the label is the target class
        other_scores: sequence of scores where the label is the non-target class
    Returns:
        eer: equal error rate
        thd: the value where the target error rate

    """
    assert len(target_scores) != 0 and len(other_scores) != 0
    target_scores_sorted = sorted(target_scores)
    other_scores_sorted = sorted(other_scores)
    target_size = float(len(target_scores_sorted))
    other_size = len(other_scores_sorted)
    target_position = 0
    for target_position, tgt_score in enumerate(target_scores_sorted[:-1]):
        other_n = other_size * target_position / target_size
        other_position = int(other_size - 1 - other_n)
        if other_position < 0:
            other_position = 0
        if other_scores_sorted[other_position] < tgt_score:
            break
    thd = target_scores_sorted[target_position]
    eer = target_position / target_size
    return eer, thd


def get_metrics(prediction, label):
    """Calculate metrics for a binary classification task.

    Args:
        prediction: sequence of probabilities
        label: sequence of class labels
    Returns:
        auc: area under curve
        eer: equal error rate
        fpr: false positive rate
        tpr: true positive rate

    """
    assert len(prediction) == len(label), (len(prediction), len(label))
    fpr, tpr, _ = metrics.roc_curve(label, prediction, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    eer, _ = compute_eer(
        [pred for i, pred in enumerate(prediction) if label[i] == 1],
        [pred for i, pred in enumerate(prediction) if label[i] == 0],
    )
    return auc, eer, fpr, tpr


def record_time(function):
    """Record running time of specific function

    Args:
        function: function to be decorated
    Returns:
        wrap: wrapped function which records its running time

    """

    def wrap():
        start = time.time()
        result = function()
        finish = time.time()
        print("Function {} runs {} seconds.".format(function.__name__, finish - start))
        return result

    return wrap


def divide_wave_to_frame(wave, frame_size=400, frame_shift=160, gate_function='rectangle'):
    """Divide audio wave into frames.

    Args:
        wave: audio wave to be divided
        frame_size: size of each frame
        frame_shift: shift of each frame
        gate_function: rectangle / hamming
    Returns:
        frames: divided frames from audio wave
        timer: corresponding time of each frame

    """
    if gate_function == 'hamming':
        gate = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(frame_size) / (frame_size - 1))
    else:
        gate = np.ones(frame_size)
    begin, length = 0, len(wave)
    frames = []
    while begin < length:
        end = begin + frame_size
        if end <= length:
            frame = wave[begin:end]
        else:
            frame = np.pad(wave[begin:], (0, end - length), 'constant', constant_values=(0, 0))
        frames.append(gate * frame)
        begin += frame_shift
    return np.array(frames, dtype=float)


def short_time_energy(frames, frame_size=400):
    """Evaluate short-time energy of each frame.

    Args:
        frames: two-dimensional array of frames generated by rectangle window
        frame_size: size of each frame
    Returns:
        energy: short-time energy of each frame

    """
    return np.sum(np.square(frames), axis=1) / frame_size


def zero_crossing_rate(frames, frame_size=400):
    """Evaluate zero-crossing rate of each frame.

    Args:
        frames: two-dimensional array of frames generated by rectangle window
        frame_size: size of each frame
    Returns:
        rate: zero-crossing rate of each frame

    """
    return np.sum(0.5 * np.abs(np.sign(frames[:, 1:]) - np.sign(frames[:, :-1])), axis=1) / frame_size


def spectral_centroid(frames, sample_rate, frame_size=400):
    """Evaluate spectral centroid of each frame.

    Args:
        frames: two-dimensional array of frames generated by hamming window
        sample_rate: sample rate of audio wave
        frame_size: size of each frame
    Returns:
        centroid: spectral centroid of each frame

    """
    magnitude = np.abs(np.fft.rfft(frames, axis=1)) / frame_size
    frequency = np.fft.rfftfreq(frame_size, 1 / sample_rate)
    return np.sum(magnitude * frequency, axis=1) / np.sum(magnitude)


def fundamental_frequency(frames, sample_rate, frame_size=400):
    """Evaluate fundamental frequency of each frame.

    Args:
        frames: two-dimensional array of frames generated by hamming window
        sample_rate: sample rate of audio wave
        frame_size: size of each frame
    Returns:
        frequency: fundamental frequency of each frame

    """
    length, low, high = len(frames), sample_rate // 2000, sample_rate // 20
    extreme = np.zeros(length)
    for i in range(length):
        relate = np.correlate(frames[i], frames[i], mode='full')[frame_size:]
        extreme[i] = low + np.argmax(relate[low:high])
    return extreme


def wave_feature(wave, sample_rate, frame_size=400, frame_shift=160):
    """Evaluate feature of given audio wave.

    Args:
        wave: audio wave to be evaluated
        sample_rate: sample rate of audio wave
        frame_size: size of each frame
        frame_shift: shift of each frame
    Returns:
        feature: two-dimensional array of wave feature, each line includes four values, which respectively
        stand for short-time energy, zero-crossing rate, spectral centroid and fundamental frequency.

    """
    frames_rectangle = divide_wave_to_frame(wave, frame_size, frame_shift, 'rectangle')
    frames_hamming = divide_wave_to_frame(wave, frame_size, frame_shift, 'hamming')
    ste = short_time_energy(frames_rectangle)
    zcr = 1e0 * zero_crossing_rate(frames_rectangle)
    spc = 1e-1 * spectral_centroid(frames_hamming, sample_rate)
    ffq = fundamental_frequency(frames_hamming, sample_rate)
    return np.stack([1e-3 * np.sqrt(ste), zcr, 1e-1 * spc, 1e-2 * ffq], axis=0).T


def mean_filtering(curve, width=10):
    """Smooth curve by mean filtering.

    Args:
        curve: audio wave to be evaluated
        width: length of filtering window
    Returns:
        curve: smooth curve generated by mean filtering

    """
    window = np.ones(width) / width
    return np.convolve(curve, window, mode='same')


def generate_prediction(probability, holder=8):
    """Convert probability to prediction by simple state machine.

    Args:
        probability: probability array of voice in each frame
        holder: minimum number of frames for state transition
    Returns:
        prediction: prediction generated by state machine

    """
    prediction = []
    active = False
    for i in range(len(probability)):
        state = probability[max(0, i-holder+1): i+1]
        if active and np.max(state) < 0.5:
            active = False
        elif not active and np.min(state) > 0.5:
            active = True
        prediction.append(int(active))
    return np.array(prediction)


def compute_estimation(probability, prediction, reality):
    """Estimate performance of specific model and plot receiver operating characteristic.

    Args:
        probability: probability array of voice in each frame
        prediction: prediction array of voice in each frame
        reality: reality array of voice in each frame
    Returns:
        acc, auc, err: accuracy, area under curve and equal error rate

    """
    acc = compute_acc(prediction, reality)
    auc, err, fpr, tpr = get_metrics(probability, reality)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.show()
    return acc, auc, err


def plot_sample(truth, prediction):
    """Plot truth and prediction as sample.

    Args:
        truth: truth array of voice in each frame
        prediction: prediction array of voice in each frame
    Returns:
        no return

    """
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(truth)
    plt.xlabel('frame')
    plt.ylabel('label')
    plt.title('truth')
    plt.subplot(2, 1, 2)
    plt.plot(prediction)
    plt.xlabel('frame')
    plt.ylabel('label')
    plt.title('prediction')
    plt.tight_layout()
    plt.show()
