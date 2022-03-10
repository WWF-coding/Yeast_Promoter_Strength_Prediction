import numpy as np
import math
from sklearn.metrics import confusion_matrix

def cal_confusion_matrix(true_label, pred_label):
    corrects = (true_label == pred_label).sum().item()
    acc = corrects / len(true_label)
    computed_confusion_matrix = confusion_matrix(true_label, pred_label)
    class_num = computed_confusion_matrix.shape[0]
    tp_num = 0
    fp_num = 0
    fn_num = 0
    tn_num = 0
    sensitivity_matrix = []
    specificity_matrix = []
    mcc_matrix = []
    for i in range(class_num):
        all_num = np.sum(computed_confusion_matrix)
        tp_num = computed_confusion_matrix[i, i]
        fp_num = np.sum(computed_confusion_matrix[:, i])-tp_num
        fn_num = np.sum(computed_confusion_matrix[i, :])-tp_num
        tn_num = all_num-tp_num-fp_num-fn_num
        sensitivity_matrix.append(tp_num/(tp_num+fn_num))
        specificity_matrix.append(tn_num/(fp_num+tn_num))
        mcc_class = (tp_num*tn_num-fp_num*fn_num)/math.sqrt((tp_num+fp_num)*(tp_num+fn_num)*(tn_num+fp_num)*(tn_num+fn_num))
        mcc_matrix.append(mcc_class)
    sensitivity = np.average(sensitivity_matrix)
    specificity = np.average(specificity_matrix)
    mcc = np.average(mcc_matrix)
    return computed_confusion_matrix, acc, sensitivity, specificity, mcc