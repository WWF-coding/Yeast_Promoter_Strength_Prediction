import os
import numpy as np
from PromoterStrengthPrediction_Fun import data_import
from sklearn.model_selection import train_test_split
from PromoterStrengthPrediction_Fun import convertSampleToPseKNC

RANDOM_STATE = 0

Dataset_save_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]+'/result'
PromoterData, StructuralProp_Di, StructuralProp_Tri = data_import()
PromoterDataset = np.load(Dataset_save_path+'/PromoterDataset_shuffle.npy')

PromoterDataset_Sequence = [seq[0] for seq in PromoterDataset]
PromoterDataset_Activity = [seq[1] for seq in PromoterDataset]

trainval_X, test_X, trainval_Y, test_Y = train_test_split(PromoterDataset_Sequence, PromoterDataset_Activity, 
    stratify=PromoterDataset_Activity, test_size=0.1, random_state=RANDOM_STATE)

trainval_X_PseKNC = convertSampleToPseKNC(trainval_X)
trainval_label = trainval_Y

test_X_PseKNC = convertSampleToPseKNC(test_X)
test_label = test_Y

from classification_algorithms import compareSVM

### PseKNC coding - model training & hyperparameters tunning ###
# grid_SVM_PseKNC = compareSVM(trainval_X_PseKNC, trainval_label, random_seed=RANDOM_STATE)
# print("The best parameters for SVM with PseKNC coding are %s with a score of %0.4f" % (grid_SVM_PseKNC.best_params_, grid_SVM_PseKNC.best_score_))

### PseKNC coding - model test ###
from sklearn.svm import SVC
from performance_evaluation import cal_confusion_matrix
from sklearn.metrics import roc_auc_score

Best_clfSVM_PseKNC = SVC(C=1, gamma=0.0001, probability=True, decision_function_shape='ovo')

Best_clfSVM_PseKNC.fit(trainval_X_PseKNC, trainval_label)
y_pred_SVM_PseKNC = Best_clfSVM_PseKNC.predict(test_X_PseKNC)
metrics_result_SVM_PseKNC = cal_confusion_matrix(test_label, y_pred_SVM_PseKNC)
y_predprob_SVM_PseKNC = Best_clfSVM_PseKNC.predict_proba(test_X_PseKNC)

predict_result = Dataset_save_path+"/y_predict_C_{}_g_{}_PseKNC.npz".format(1, 0.0001)
test_trues = test_label
test_preds = y_predprob_SVM_PseKNC
np.savez_compressed(predict_result, test_predsls=test_preds, test_truesls=test_trues)

print("Accuracy for SVM with PseKNC coding: %0.4f" % (metrics_result_SVM_PseKNC[1]))
print("Sensitivity for SVM with PseKNC coding: %0.4f" % (metrics_result_SVM_PseKNC[2]))
print("Specificity for SVM with PseKNC coding: %0.4f" % (metrics_result_SVM_PseKNC[3]))