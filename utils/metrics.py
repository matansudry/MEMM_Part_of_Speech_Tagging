import pandas as pd

def accuracy(pred_tags, true_tags):
    correct = 0
    total = 0
    for preds, tags in zip(pred_tags, true_tags):
        if isinstance(preds, list):
            for pred, tag in zip(preds, tags):
                total += 1
                if pred == tag:
                    correct += 1
        else:
            total += 1
            if pred == tag:
                correct += 1

    if total > 0:
        return float(correct)/total
    return 0.0


def confusion_matrix(tags, pred_tags, true_tags):
    """
    args:
        * tags
        * pred_tags
        * true_tags
    returns:
        * confusion_matrix - rows are true_tags, columns are pred_tags
        * worst_tags - dict of key = tag, val = 1 vs all accuracy, sorted from worst to best
    """
    matrix = pd.DataFrame(0, index=sorted(tags), columns=sorted(tags))
    for preds, trues in zip(pred_tags, true_tags):
        if isinstance(preds, list):
            for pred, tag in zip(preds, trues):
                matrix[pred][tag] += 1
        else:
            matrix[preds][trues] += 1
    
    worst_tags = {}
    matrix_sum = matrix.sum().sum()
    for tag in tags:
        tags_wo_tag = set(tags).difference({tag})
        TP = matrix[tag][tag]
        TN = matrix[tags_wo_tag].loc[tags_wo_tag].sum().sum()
        worst_tags[tag] = (TP + TN)/matrix_sum

    return matrix, dict(sorted(list(worst_tags.items()), key=lambda x: x[1], reverse=False))


