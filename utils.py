import numpy as np

'''
Have taken this code from a kaggle blog
'''
def precision_at_k(y_true, y_pred, k=12):
    """ Computes Precision at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Precision at k
    """
    intersection = np.intersect1d(y_true, y_pred[:k])
    return len(intersection) / k


def rel_at_k(y_true, y_pred, k=12):
    """ Computes Relevance at k for one sample

    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations

    Returns
    _______
    score: double
           Relevance at k
    """
    if y_pred[k-1] in y_true:
        return 1
    else:
        return 0


def average_precision_at_k(y_true, y_pred, k=12):
    """ Computes Average Precision at k for one sample
    
    Parameters
    __________
    y_true: np.array
            Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           Average Precision at k
    """
    ap = 0.0
    for i in range(1, k+1):
        ap += precision_at_k(y_true, y_pred, i) * rel_at_k(y_true, y_pred, i)
        
    return ap / min(k, len(y_true))


def mean_average_precision(y_true, y_pred, k=12):
    """ Computes MAP at k
    
    Parameters
    __________
    y_true: np.array
            2D Array of correct recommendations (Order doesn't matter)
    y_pred: np.array
            2D Array of predicted recommendations (Order does matter)
    k: int, optional
       Maximum number of predicted recommendations
            
    Returns
    _______
    score: double
           MAP at k
    """
    return np.mean([average_precision_at_k(gt, pred, k) \
                    for gt, pred in zip(y_true, y_pred)])
    


def calculate_map_at_k(gt, pred, k=5):
    assert len(gt) == len(pred), "The length of ground truth and predictions should be the same"
    
    num_queries = len(gt)
    average_precisions = []
    
    for i in range(num_queries):
        gt_set = set(gt[i])
        pred_list = pred[i][:k]
        num_relevant = 0
        precisions = []
        
        for j in range(len(pred_list)):
            if pred_list[j] in gt_set:
                num_relevant += 1
                precisions.append(num_relevant / (j + 1))
        
        if num_relevant == 0:
            average_precisions.append(0)
        else:
            average_precisions.append(sum(precisions) / num_relevant)
    
    return np.mean(average_precisions)

