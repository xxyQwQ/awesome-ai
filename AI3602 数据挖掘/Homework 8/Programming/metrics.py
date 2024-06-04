import math


def metric_acc(real, predict):
    """XXX: Do NOT modify this function."""
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1
    return sum / float(len(real))


def metric_mrr(real, predict):
    """XXX: Do NOT modify this function."""
    sum = 0.0
    for val in real:
        try:
            index = predict.index(val)
        except ValueError:
            index = -1
        if index != -1:
            sum = sum + 1.0 / float(index + 1)
    return sum / float(len(real))


def metric_ndcg(real, predict):
    """XXX: Do NOT modify this function."""
    dcg = 0.0
    idcg = metric_idcg(len(real))
    for i, predictItem in enumerate(predict):
        if predictItem in real:
            itemRelevance = 1
            rank = i + 1
            dcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(rank + 1))
    return dcg / float(idcg)


def metric_idcg(n):
    """XXX: Do NOT modify this function."""
    idcg = 0
    itemRelevance = 1
    for i in range(n):
        idcg += (math.pow(2, itemRelevance) - 1.0) * (math.log(2) / math.log(i + 2))
    return idcg
