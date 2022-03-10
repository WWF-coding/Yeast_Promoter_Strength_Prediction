import random
import numpy as np
import math
from itertools import product
import os
import csv
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import shuffle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import glob
from pytorchtools import EarlyStopping

from PromoterStrengthPrediction_Fun import data_import
from PromoterStrengthPrediction_Fun import convertSampleToOneHot_MonoMer
from PromoterStrengthPrediction_Fun import convertSampleToOneHot_TriMer
from PromoterStrengthPrediction_Fun import convertSampleToDNC
from PromoterStrengthPrediction_Fun import convertSampleToPhyChemVector_Di
from PromoterStrengthPrediction_Fun import convertSampleToPhyChemVector_Tri
from PromoterStrengthPrediction_Fun import MyDataset, MyDataset_DNC
from Models import BranchedCNN_Net, iPSW_PseDNC_DL_Net

RANDOM_STATE = 0
torch.manual_seed(RANDOM_STATE)

TOTAL_EPOCHES = 300
DROPOUT_PROB = 0.28
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001

Dataset_save_path = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]+'/result'
FILE_MODEL_TMP = "model_tmp.pkl"
MODEL_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]+'/result/model_seed' + str(RANDOM_STATE)
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class Regularization(nn.Module):
    def __init__(self, model, weight_decay, p=2):
        super(Regularization, self).__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
    
    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        
        return reg_loss

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss+l2_reg
        reg_loss = weight_decay*reg_loss
        
        return reg_loss

def loss_function(pred_label, true_label, model, weightdecay):
    criterion = nn.CrossEntropyLoss()
    reg_loss = Regularization(model, weight_decay=weightdecay, p=2)
    loss = criterion(pred_label, true_label.long())+reg_loss(model)

    return loss

def train_one_epoch(model, train_loader, learning_rate, weightdecay):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_loss_epoch = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs_I, inputs_II, inputs_III, inputs_IV, labels = data
        #inputs_I, inputs_DNC, labels = data #
        optimizer.zero_grad()
        output = model(inputs_I, inputs_II, inputs_III, inputs_IV)
        # output = model(inputs_I, inputs_DNC) #
        loss = loss_function(output, labels, model, weightdecay)
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item()
    train_loss_avg_epoch = train_loss_epoch/len(train_loader)
    return train_loss_avg_epoch

def evaluate(file_model, loader, dropoutprob):
    model = BranchedCNN_Net(dropout_prob=dropoutprob)
    # model = iPSW_PseDNC_DL_Net(dropout_prob=dropoutprob)
    model.load_state_dict(torch.load(file_model))
    model.eval()
    criterion = nn.CrossEntropyLoss()
    eval_loss_epoch = 0.0
    for i, data in enumerate(loader, 0):
        inputs_I, inputs_II, inputs_III, inputs_IV, labels = data
        # inputs_I, inputs_DNC, labels = data #
        output = model(inputs_I, inputs_II, inputs_III, inputs_IV)
        # output = model(inputs_I, inputs_DNC) #
        pred_prob = output.detach().numpy()
        loss = criterion(output, labels.long())
        eval_loss_epoch += loss.item()
    eval_loss_avg_epoch = eval_loss_epoch/len(loader)
    _, pred_label = output.max(1)
    _, acc, sensitivity, specificity = cal_confusion_matrix(labels, pred_label)
    result = {'eval_loss_avg_epoch': eval_loss_avg_epoch,
                'confusion_matrix': confusion_matrix,
                'acc': acc,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'pred_prob': pred_prob,
                'true label': labels,
                'pred_label': pred_label
                }
    return result

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
    sensitivity = np.average(sensitivity_matrix)
    specificity = np.average(specificity_matrix)
    return computed_confusion_matrix, acc, sensitivity, specificity

def train_one_fold(fold, train_loader, val_loader, dropoutprob, weightdecay, learning_rate):
    model = BranchedCNN_Net(dropout_prob=dropoutprob)
    # model = iPSW_PseDNC_DL_Net(dropout_prob=dropoutprob) #
    model.apply(weights_init)

    best_epoch = 0
    file_model = "best_model_saved.pkl"

    train_loss = []
    val_loss = []

    patience = 30
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    for epoch in range(TOTAL_EPOCHES):
        train_loss_avg_epoch = train_one_epoch(model, train_loader, learning_rate, weightdecay)
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, FILE_MODEL_TMP))

        file_model_tmp = os.path.join(MODEL_DIR, FILE_MODEL_TMP)
        result_val = evaluate(file_model_tmp, val_loader, dropoutprob)
        valid_loss_avg_epoch = result_val['eval_loss_avg_epoch']
        print('Fold: {}, Epoch: {}/{}, train_loss: {:.4f}, val_loss: {:.4f}'.format(fold+1, epoch+1, TOTAL_EPOCHES, 
            train_loss_avg_epoch, valid_loss_avg_epoch))
        train_loss.append(train_loss_avg_epoch)
        val_loss.append(valid_loss_avg_epoch)

        early_stopping(valid_loss_avg_epoch, model)
        best_epoch = epoch-patience
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    model.load_state_dict(torch.load('checkpoint.pkl'))
    model_fn = "promoter_{}_{}.pkl".format(fold, best_epoch)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, model_fn))

    with open(MODEL_DIR+"/logfile_loss_model_{}_{}.csv".format(fold, best_epoch), mode = 'w') as lf_loss:
        lf_loss = csv.writer(lf_loss, delimiter=',')
        lf_loss.writerow(['epoch', 'train loss', 'valid loss'])
        for i in range(len(train_loss)):
            lf_loss.writerow([i, train_loss[i], val_loss[i]])
    
    file_model = os.path.join(MODEL_DIR, model_fn)
    result_val = evaluate(file_model, val_loader, dropoutprob)
    
    return result_val['eval_loss_avg_epoch']

def train_kfold(dropoutprob, weightdecay, learning_rate):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold = 0
    CV_valid_loss = []
    for train_index, val_index in skf.split(trainval_data, trainval_Y):
        train_X_OHM, val_X_OHM = trainval_X_OHM[train_index], trainval_X_OHM[val_index]
        train_X_OHT, val_X_OHT = trainval_X_OHT[train_index], trainval_X_OHT[val_index]
        train_X_DNC, val_X_DNC = trainval_X_DNC[train_index], trainval_X_DNC[val_index]
        train_X_PCVD, val_X_PCVD = trainval_X_PCVD[train_index], trainval_X_PCVD[val_index]
        train_X_PCVT, val_X_PCVT = trainval_X_PCVT[train_index], trainval_X_PCVT[val_index]
        train_Y, val_Y = np.array(trainval_Y)[train_index], np.array(trainval_Y)[val_index]
        
        train_data = MyDataset(train_X_OHM, train_X_OHT, train_X_PCVD, train_X_PCVT, train_Y)
        # train_data = MyDataset_DNC(train_X_OHM, train_X_DNC, train_Y) #
        val_data = MyDataset(val_X_OHM, val_X_OHT, val_X_PCVD, val_X_PCVT, val_Y)
        # val_data = MyDataset_DNC(val_X_OHM, val_X_DNC, val_Y) #

        train_loader = DataLoader(train_data, batch_size=42, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=42)

        valid_loss_fold = train_one_fold(fold, train_loader, val_loader, dropoutprob, weightdecay, learning_rate)
        CV_valid_loss.append(valid_loss_fold)
        fold += 1

    return np.mean(CV_valid_loss)

def hyperpara_tuning():
    Best_val_loss = float('inf')
    parameters = dict(
        dropout_prob = [0.1, 0.15, 0.21, 0.28, 0.36],
        learning_rate = [0.1, 0.01, 0.001],
    )
    param_values = [v for v in parameters.values()]
    for dropoutprob, learningrate in product(*param_values):
        print("### DROPOUT_PROB = {}, LEARNING_RATE = {} ###".format(dropoutprob, learningrate))
        val_loss = train_kfold(dropoutprob, WEIGHT_DECAY, learningrate)
        if val_loss < Best_val_loss:
            Best_val_loss = val_loss
            Best_dropoutprob = dropoutprob
            Best_learningrate = learningrate
    print("Best_val_loss = {:.4f}, Best_dropoutprob = {}, Best_learningrate = {}".format(Best_val_loss, 
        Best_dropoutprob, Best_learningrate))

def test():
    with open(trainvalresult_fn, mode='w') as outfile:
        outfile = csv.writer(outfile, delimiter=',')
        outfile.writerow(['model_fn', 'Accuracy', 'Sensitivity', 'Specificity'])
    with open(testresult_fn, mode='w') as outfile:
        outfile = csv.writer(outfile, delimiter=',')
        outfile.writerow(['model_fn', 'Accuracy', 'Sensitivity', 'Specificity'])

    list_model_fn = sorted(glob.glob(MODEL_DIR+"/promoter_*.pkl"))

    y_prob_mtx_trainval = []
    y_prob_mtx_test = []

    for model_fn in list_model_fn:
        result_trainval = evaluate(model_fn, trainval_loader, dropoutprob=DROPOUT_PROB)
        result_test = evaluate(model_fn, test_loader, dropoutprob=DROPOUT_PROB)
        y_prob_mtx_trainval.append(result_trainval['pred_prob'])
        y_prob_mtx_test.append(result_test['pred_prob'])

        with open(trainvalresult_fn, mode='a') as outfile:
            outfile = csv.writer(outfile, delimiter=',')
            outfile.writerow([model_fn, result_trainval['acc'], result_trainval['sensitivity'], result_trainval['specificity']])
        with open(testresult_fn, mode='a') as outfile:
            outfile = csv.writer(outfile, delimiter=',')
            outfile.writerow([model_fn, result_test['acc'], result_test['sensitivity'], result_test['specificity']])
    
    y_prob_mtx_trainval = np.array(y_prob_mtx_trainval)
    y_prob_ensemble_trainval = [np.mean(y_prob_mtx_trainval[:,row], axis=0) for row in range(np.size(y_prob_mtx_trainval, 1))]
    y_prob_ensemble_trainval = np.array(y_prob_ensemble_trainval)
    y_pred_ensemble_trainval = np.argmax(y_prob_ensemble_trainval, axis=1)
    y_true_trainval = trainval_Y
    _, acc_ensemble_trainval, sensitivity_ensemble_trainval, specificity_ensemble_trainval = cal_confusion_matrix(y_true_trainval, 
        y_pred_ensemble_trainval)
    print("Metrics on Trainval Set: Acc = {:.4f}, Sn = {:.4f}, Sp = {:.4f}".format(acc_ensemble_trainval, 
        sensitivity_ensemble_trainval, specificity_ensemble_trainval))
    with open(trainvalresult_fn, mode='a') as outfile:
        outfile = csv.writer(outfile, delimiter=',')
        outfile.writerow(["ensemble", acc_ensemble_trainval, sensitivity_ensemble_trainval, specificity_ensemble_trainval])

    predict_result = Dataset_save_path + '/y_predict_P_{}_LR_{}_ALL.npz'.format(DROPOUT_PROB, LEARNING_RATE)
    y_prob_mtx_test = np.array(y_prob_mtx_test)
    y_prob_ensemble_test = [np.mean(y_prob_mtx_test[:,row], axis=0) for row in range(np.size(y_prob_mtx_test, 1))]
    y_prob_ensemble_test = np.array(y_prob_ensemble_test)
    y_pred_ensemble_test = np.argmax(y_prob_ensemble_test, axis=1)
    y_true_test = test_Y
    test_trues = y_true_test
    test_predicts = y_prob_ensemble_test
    np.savez_compressed(predict_result, test_truesls=test_trues, test_predsls=test_predicts)
    _, acc_ensemble_test, sensitivity_ensemble_test, specificity_ensemble_test = cal_confusion_matrix(y_true_test, y_pred_ensemble_test)
    print("Metrics on Test Set: Acc = {:.4f}, Sn = {:.4f}, Sp = {:.4f}".format(acc_ensemble_test, 
        sensitivity_ensemble_test, specificity_ensemble_test))
    with open(testresult_fn, mode='a') as outfile:
        outfile = csv.writer(outfile, delimiter=',')
        outfile.writerow(["ensemble", acc_ensemble_test, sensitivity_ensemble_test, specificity_ensemble_test])
    
if __name__ == "__main__":
    ## Import dataset ###
    PromoterData, StructuralProp_Di, StructuralProp_Tri = data_import()

    PromoterDataset = np.load(Dataset_save_path+'/PromoterDataset_shuffle.npy')

    PromoterDataset_Sequence = [seq[0] for seq in PromoterDataset]
    PromoterDataset_Activity = [int(seq[1]) for seq in PromoterDataset]

    ### Split dataset ###
    trainval_X, test_X, trainval_Y, test_Y = train_test_split(PromoterDataset_Sequence, PromoterDataset_Activity, 
        stratify=PromoterDataset_Activity, test_size=0.1, random_state=RANDOM_STATE)

    ### Data formulation ###
    trainval_X_OHM = convertSampleToOneHot_MonoMer(trainval_X)
    trainval_X_OHT = convertSampleToOneHot_TriMer(trainval_X)
    trainval_X_DNC = convertSampleToDNC(trainval_X)
    trainval_X_PCVD = convertSampleToPhyChemVector_Di(trainval_X, StructuralProp_Di)
    trainval_X_PCVT = convertSampleToPhyChemVector_Tri(trainval_X, StructuralProp_Tri)

    test_X_OHM = convertSampleToOneHot_MonoMer(test_X)
    test_X_OHT = convertSampleToOneHot_TriMer(test_X)
    test_X_DNC = convertSampleToDNC(test_X)
    test_X_PCVD = convertSampleToPhyChemVector_Di(test_X, StructuralProp_Di)
    test_X_PCVT = convertSampleToPhyChemVector_Tri(test_X, StructuralProp_Tri)
    
    ### Load data ###
    trainval_data = MyDataset(trainval_X_OHM, trainval_X_OHT, trainval_X_PCVD, trainval_X_PCVT, trainval_Y)
    # trainval_data = MyDataset_DNC(trainval_X_OHM, trainval_X_DNC, trainval_Y) #
    test_data = MyDataset(test_X_OHM, test_X_OHT, test_X_PCVD, test_X_PCVT, test_Y)
    # test_data = MyDataset_DNC(test_X_OHM, test_X_DNC, test_Y) #
    trainval_loader = DataLoader(trainval_data, batch_size=len(trainval_Y))
    test_loader = DataLoader(test_data, batch_size=len(test_Y))

    ### Parameter tuning ###
    # hyperpara_tuning()

    trainvalresult_fn = MODEL_DIR+"/trainval_result.csv"
    testresult_fn = MODEL_DIR+"/test_result.csv"

    ### model training ###
    # train_kfold(dropoutprob=DROPOUT_PROB, weightdecay=WEIGHT_DECAY, learning_rate=LEARNING_RATE)

    ### model test ###
    test()