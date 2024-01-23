class Evaluator():

    @staticmethod
    def accuracy(predictions, labels):
        corr, total = 0, 0
        for i, pred in enumerate(predictions):
            total += 1
            corr += set(pred) == set(labels[i])
        return 100 * corr / total

    @staticmethod
    def fscore(predictions, labels):
        TP, TP_FP, TP_FN = 0, 0, 0
        for i in range(len(predictions)):
            pred = set(predictions[i])
            label = set(labels[i])
            TP += len(pred & label)
            TP_FP += len(pred)
            TP_FN += len(label)
        if TP_FP == 0:
            precision = 0
        else:
            precision = TP / TP_FP
        recall = TP / TP_FN
        if precision + recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        return {'precision': 100 * precision, 'recall': 100 * recall, 'fscore': 100 * fscore}

    def acc(self, predictions, labels):
        metric_dicts = {}
        metric_dicts['acc'] = self.accuracy(predictions, labels)
        metric_dicts['fscore'] = self.fscore(predictions, labels)
        return metric_dicts
