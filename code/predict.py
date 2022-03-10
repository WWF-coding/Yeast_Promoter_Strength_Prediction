import numpy as np
import torch 
import glob
import os
from torch.utils.data import DataLoader
from Models import BranchedCNN_Net

from PromoterStrengthPrediction_Fun import data_import
from PromoterStrengthPrediction_Fun import convertSampleToOneHot_MonoMer
from PromoterStrengthPrediction_Fun import convertSampleToOneHot_TriMer
from PromoterStrengthPrediction_Fun import convertSampleToPhyChemVector_Di
from PromoterStrengthPrediction_Fun import convertSampleToPhyChemVector_Tri
from PromoterStrengthPrediction_Fun import MyDataset_pred

MODEL_DIR = os.path.dirname(os.getcwd())+'/result/model_seed0'

def predict(file_model, inputs_loader):
    model = BranchedCNN_Net()
    model.load_state_dict(torch.load(file_model))
    model.eval()
    for i, data in enumerate(inputs_loader, 0):
        inputs_I, inputs_II, inputs_III, inputs_IV = data
        output = model(inputs_I, inputs_II, inputs_III, inputs_IV)
        pred_prob = output.detach().numpy()
    return pred_prob

if __name__ == '__main__':

    _, StructuralProp_Di, StructuralProp_Tri = data_import()

    inputs_Sequence = [
        'ATCAAAATTTAACTGTTCTAACCCCTACTTGACAGCAATATATAAACAGAAGGAAGCTGCCCTGTCTTAAACCTTTTTTTTTATCATCAT'
        ]

    inputs_X_OHM = convertSampleToOneHot_MonoMer(inputs_Sequence)
    inputs_X_OHT = convertSampleToOneHot_TriMer(inputs_Sequence)
    inputs_X_PCVD = convertSampleToPhyChemVector_Di(inputs_Sequence, StructuralProp_Di)
    inputs_X_PCVT = convertSampleToPhyChemVector_Tri(inputs_Sequence, StructuralProp_Tri)

    ### Load data ###
    inputs_data = MyDataset_pred(inputs_X_OHM, inputs_X_OHT, inputs_X_PCVD, inputs_X_PCVT)
    inputs_loader = DataLoader(inputs_data, batch_size=len(inputs_data))

    list_model_fn = sorted(glob.glob(MODEL_DIR+"/promoter_*.pkl"))

    y_prob_mtx_pred = []

    for model_fn in list_model_fn:
        y_prob_pred = predict(model_fn, inputs_loader)
        y_prob_mtx_pred.append(y_prob_pred)

    y_prob_mtx_pred = np.array(y_prob_mtx_pred)
    y_prob_ensemble_pred = [np.mean(y_prob_mtx_pred[:,row], axis=0) for row in range(np.size(y_prob_mtx_pred, 1))]
    y_prob_ensemble_pred = np.array(y_prob_ensemble_pred)
    y_pred_ensemble = np.argmax(y_prob_ensemble_pred, axis=1)
    
    print("Promoters class label", y_pred_ensemble)