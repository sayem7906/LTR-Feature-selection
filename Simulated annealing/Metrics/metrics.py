import math
import numpy as np
from pyltr.util.sort import get_sorted_y
from sklearn.metrics import ndcg_score,average_precision_score as ap
from sklearn.preprocessing import normalize
def ndcg_p(ordered_data, p):
    """normalised discounted cumulative gain"""
    if sum(ordered_data)==0:
        return 0
    else:
        indexloop = range(0, p)
        DCG_p = 0
        for index in indexloop:
            current_ratio=(2**(ordered_data[index])-1)*(math.log((float(index)+2), 2)**(-1))
            DCG_p = DCG_p + current_ratio
        ordered_data.sort(reverse=True) 
        # ordered_data = sorted(ordered_data,reverse=True)
        indexloop = range(0, p)
        iDCG_p = 0
        for index in indexloop:
            current_ratio=(2**(ordered_data[index])-1)*((math.log((index+2), 2))**(-1))
            iDCG_p = iDCG_p + current_ratio
        return(DCG_p/iDCG_p)
    
    
    
def average_precision_score(y_true, y_score, k=10):
    """Average precision at rank k
    Parameters
    ----------
    y_true : array-like, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array-like, shape = [n_samples]
        Predicted scores.
    k : int
        Rank.
    Returns
    -------
    average precision @k : float
    """
    unique_y = np.unique(y_true)

    if len(unique_y) > 2:
        raise ValueError("Only supported for two relevance levels.")

    pos_label = 1
    n_pos = np.sum(y_true == pos_label)

    # order = np.argsort(y_score)[::-1][:min(n_pos, k)]
    # y_true = np.asarray(y_true)[order]
    y_true = [x for _,x in sorted(zip(y_score,y_true), reverse=True)][:min(n_pos, k)]

    score = 0
    for i in range(len(y_true)):
        if y_true[i] == pos_label:
            # Compute precision up to document i
            # i.e, percentage of relevant documents up to document i.
            prec = 0
            for j in range(0, i + 1):
                if y_true[j] == pos_label:
                    prec += 1.0
            prec /= (i + 1.0)
            score += prec

    if n_pos == 0:
        return 0

    return score / n_pos



def ndcg_mean(group_test,y_test,pred,k=10):
        nquerys=range(0,len(group_test))
        lower=0
        upper=0
        ndcgs=[]
        for i in nquerys:
                many=group_test[i]
                upper = upper+many
                if many>1:
                    y_pred = np.reshape(pred[lower:upper],(1,-1))
                    y_true = np.reshape(y_test[lower:upper].astype(int),(1,-1))
                    
                    # ordered = [x for _,x in sorted(zip(y_pred,y_true), reverse=True)]
                    # ordered = get_sorted_y(y_true, y_pred)
                    # print(ordered)
                    if(many<k):
                        temp = many
                    else:
                        temp=k
                    # result = ndcg_p(ordered, temp)
                    result = ndcg_score(y_true, y_pred,k=temp)
                else:
                    result = 1
                ndcgs.append(result)
                lower=upper
        # ndcgs = sorted(ndcgs,reverse=True)
        # print(ndcgs)
        # print('\n')
        return np.mean(ndcgs)
def mean_ap(group_test,y_test,pred,k=10):
        nquerys=range(0,len(group_test))
        lower=0
        upper=0
        aps=[]
        for i in nquerys:
                many=group_test[i]
                upper = upper+many
    
                y_pred = pred[lower:upper]
                y_true = y_test[lower:upper]
                
                mx_true = np.max(y_true)
                if(mx_true):
                    y_true = y_true/2
                y_true = np.asarray(y_true>=.5,dtype=int)
                
                result = average_precision_score(y_true, y_pred,k)
                # print(result)
                aps.append(result)
                lower=upper
        
        aps = np.nan_to_num(aps)
        # print(aps)
        return np.mean(aps)