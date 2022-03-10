import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
import random
import numpy as np
import glob
import torch 
from torch.utils.data import DataLoader
from Models import BranchedCNN_Net

from PromoterStrengthPrediction_Fun import data_import
from PromoterStrengthPrediction_Fun import convertSampleToOneHot_MonoMer
from PromoterStrengthPrediction_Fun import convertSampleToOneHot_TriMer
from PromoterStrengthPrediction_Fun import convertSampleToPhyChemVector_Di
from PromoterStrengthPrediction_Fun import convertSampleToPhyChemVector_Tri
from PromoterStrengthPrediction_Fun import MyDataset_pred

MODEL_DIR = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]+'/result/model_seed0'
DEVICE_NUM = 1

class Chromosome:
    def __init__(self, chr_base, chr_len):

        self.chr_base = chr_base
        self.chr_len = chr_len
        self.Chr = []
        self.TA_content = 0
        self.y = 0

        self.rand_init()
        self.func()

    def rand_init(self):

        # tmp = ""
        # for _ in range(self.chr_len):
        #     tmp += random.choice('ATCG')
        # self.Chr.append(tmp)
        self.Chr.append(self.chr_base)
        self.TA_content = self.chr_base[:70].count('T')/70+self.chr_base[70:].count('A')/20
               
    def predict(self, file_model, inputs_loader):
        model = BranchedCNN_Net()
        model.load_state_dict(torch.load(file_model))
        # model = torch.load(file_model, map_location=torch.device('cpu'))
        # model.to(device=torch.device('cuda:%d' % DEVICE_NUM))
        model.eval()
        for i, data in enumerate(inputs_loader, 0):
            inputs_I, inputs_II, inputs_III, inputs_IV = data
            output = model(inputs_I, inputs_II, inputs_III, inputs_IV)
            pred_prob = output.detach().numpy()
        return pred_prob

    def func(self):

        _, StructuralProp_Di, StructuralProp_Tri = data_import()
        inputs_X_OHM = convertSampleToOneHot_MonoMer(self.Chr)
        inputs_X_OHT = convertSampleToOneHot_TriMer(self.Chr)
        inputs_X_PCVD = convertSampleToPhyChemVector_Di(self.Chr, StructuralProp_Di)
        inputs_X_PCVT = convertSampleToPhyChemVector_Tri(self.Chr, StructuralProp_Tri)

        ### Load data ###
        inputs_data = MyDataset_pred(inputs_X_OHM, inputs_X_OHT, inputs_X_PCVD, inputs_X_PCVT)
        inputs_loader = DataLoader(inputs_data, batch_size=len(inputs_data))

        list_model_fn = sorted(glob.glob(MODEL_DIR+"/promoter_*.pkl"))

        y_prob_mtx_pred = []

        for model_fn in list_model_fn:
            y_prob_pred = self.predict(model_fn, inputs_loader)
            y_prob_mtx_pred.append(y_prob_pred)

        y_prob_mtx_pred = np.array(y_prob_mtx_pred)
        y_prob_ensemble_pred = [np.mean(y_prob_mtx_pred[:,row], axis=0) for row in range(np.size(y_prob_mtx_pred, 1))]
        y_prob_ensemble_pred = np.array(y_prob_ensemble_pred)
        self.label = np.argmax(y_prob_ensemble_pred, axis=1)
        self.y = self.TA_content

if __name__ == '__main__':

    chr_base = 'ATCAAAATTTAACTGTTCTAACCCCTACTTGACAGCAATATATAAACAGAAGGAAGCTGCCCTGTCTTAAACCTTTTTTTTTATCATCAT'
    chr_len = 90
    chromosome = Chromosome(chr_base, chr_len)
    print(chromosome.Chr[0])
    print(chromosome.label, chromosome.TA_content)