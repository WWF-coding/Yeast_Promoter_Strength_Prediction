def data_import():
    import os, sys
    import xlrd
    file1 = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]+'/data/PromoterSequence_Strength.xlsx'
    file2 = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]+'/data/Physicochemical_properties_Di.xlsx'
    file3 = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]+'/data/Physicochemical_properties_Tri.xlsx'
    data1 = xlrd.open_workbook(filename=file1)
    data2 = xlrd.open_workbook(filename=file2)
    data3 = xlrd.open_workbook(filename=file3)
    data1 = data1.sheet_by_index(0)
    data2 = data2.sheet_by_index(0)
    data3 = data3.sheet_by_index(0)

    PromoterData = []
    for i in range(1, data1.nrows):
        rows = data1.row_values(i)
        PromoterData.append(rows)
    StructuralProp_Di = {}
    for i in range(1, data2.nrows):
        rows = data2.row_values(i)
        StructuralProp_Di[rows[0]] = rows[1:]
    StructuralProp_Tri = {}
    for i in range(1, data3.nrows):
        rows = data3.row_values(i)
        StructuralProp_Tri[rows[0]] = rows[1:]    
    
    return PromoterData, StructuralProp_Di, StructuralProp_Tri

def data_preprocessed(sample):

    TSS = 201
    PromoterDataset = []
    for i in range(len(sample)):
        activity = sample[i][1]
        if activity >= 0.4:     # high
            activity = 2 
        elif activity >= 0.1:    # medium
            activity = 1
        else:                    # low
            activity = 0
        sequence = sample[i][2]
        trun_sequence = sequence[TSS-70:TSS+20]
        PromoterDataset.append([trun_sequence, activity])
    
    return PromoterDataset

def convertSampleToOneHot_MonoMer(sampleSeq):
    import numpy as np

    MonomerDict = {}
    MonomerDict['A'] = np.c_[1, 0, 0, 0]
    MonomerDict['T'] = np.c_[0, 1, 0, 0]
    MonomerDict['C'] = np.c_[0, 0, 1, 0]
    MonomerDict['G'] = np.c_[0, 0, 0, 1]
    NucleotideCategory = 4
    convertMatr = np.zeros([len(sampleSeq), NucleotideCategory, len(sampleSeq[0])])
    sampleNo = 0
    for sequence in sampleSeq:
        NucleotideNo = 0
        for Nucleotide in sequence:
            convertMatr[sampleNo][:, NucleotideNo] = MonomerDict[Nucleotide]
            NucleotideNo += 1
        sampleNo += 1
    
    return convertMatr

def convertSampleToOneHot_DiMer(sampleSeq):
    import numpy as np
    import operator

    DimerDict = {}
    DimerDict['AA'] = np.c_[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    DimerDict['AT'] = np.c_[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    DimerDict['AC'] = np.c_[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    DimerDict['AG'] = np.c_[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    DimerDict['TA'] = np.c_[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    DimerDict['TT'] = np.c_[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    DimerDict['TC'] = np.c_[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    DimerDict['TG'] = np.c_[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    DimerDict['CA'] = np.c_[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    DimerDict['CT'] = np.c_[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    DimerDict['CC'] = np.c_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    DimerDict['CG'] = np.c_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    DimerDict['GA'] = np.c_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    DimerDict['GT'] = np.c_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    DimerDict['GC'] = np.c_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    DimerDict['GG'] = np.c_[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    DiNucleotideCategory = 16
    convertMatr = np.zeros([len(sampleSeq), DiNucleotideCategory, len(sampleSeq[0])-1])
    sampleNo = 0
    for sequence in sampleSeq:
        DiNucleotideNo = 0
        for DiNucleotide in map(operator.add, sequence[::1], sequence[1::1]):
            convertMatr[sampleNo][:, DiNucleotideNo] = DimerDict[DiNucleotide]
            DiNucleotideNo += 1
        sampleNo += 1
    
    return convertMatr

def convertSampleToOneHot_TriMer(sampleSeq):
    import numpy as np
    import operator

    base_list = 'ATCG'
    trimers_list = [x+y+z for x in base_list for y in base_list for z in base_list]
    convertMatr = np.zeros([len(sampleSeq), len(trimers_list), len(sampleSeq[0])-2])
    sampleNo = 0
    for sequence in sampleSeq:
        TriNucleotideNo = 0
        for TriNucleotide in map(operator.add, map(operator.add, sequence[::1], sequence[1::1]), sequence[2::1]):
            letter = [0 for _ in range(64)]
            letter[trimers_list.index(TriNucleotide)] = 1
            convertMatr[sampleNo][:, TriNucleotideNo] = letter
            TriNucleotideNo += 1
        sampleNo += 1
    
    return convertMatr

def convertSampleToDNC(sampleSeq):
    import numpy as np
    import operator

    base_list = 'ATCG'
    dimers_list = [x+y for x in base_list for y in base_list]
    convertMatr = np.zeros([len(sampleSeq), 1, len(dimers_list)])
    sampleNo = 0
    for sequence in sampleSeq:
        for DiNucleotide in map(operator.add, sequence[::1], sequence[1::1]):
            letter = [0 for _ in range(16)]
            letter[dimers_list.index(DiNucleotide)] = 1
            convertMatr[sampleNo][0, :] += letter
        sampleNo += 1
    
    return convertMatr

def convertSampleToPseKNC(sampleSeq):
    import numpy as np

    MonomerDict_PseKNC = {}
    MonomerDict_PseKNC['A'] = [1, 1, 1]
    MonomerDict_PseKNC['T'] = [0, 0, 1]
    MonomerDict_PseKNC['C'] = [0, 1, 0]
    MonomerDict_PseKNC['G'] = [1, 0, 0]

    convertMatr = np.zeros([len(sampleSeq), len(sampleSeq[0])*4])
    sampleNo = 0
    for sequence in sampleSeq:
        for i in range(len(sequence)):
            Nucleotide = sequence[i]
            P = sequence[:i+1].count(Nucleotide)/(i+1)
            convertMatr[sampleNo][i*4:i*4+3] = MonomerDict_PseKNC[Nucleotide]
            convertMatr[sampleNo][i*4+3] = P
        sampleNo += 1
    
    return convertMatr

def convertSampleToPhyChemVector_Di(sampleSeq, PhyChemPropTable_Di):
    import numpy as np
    import operator

    DiNucleotidePhyChemPropNum = 12
    convertMatr = np.zeros([len(sampleSeq), DiNucleotidePhyChemPropNum, len(sampleSeq[0])-1])
    sampleNo = 0
    for sequence in sampleSeq:
        DiNucleotideNo = 0
        for DiNucleotide in map(operator.add, sequence[::1], sequence[1::1]):
            convertMatr[sampleNo][:, DiNucleotideNo] = PhyChemPropTable_Di[DiNucleotide]
            DiNucleotideNo += 1
        sampleNo += 1
    
    return convertMatr

def convertSampleToPhyChemVector_Tri(sampleSeq, PhyChemPropTable_Tri):
    import numpy as np
    import operator

    TriNucleotidePhyChemPropNum = 12
    convertMatr = np.zeros([len(sampleSeq), TriNucleotidePhyChemPropNum, len(sampleSeq[0])-2])
    sampleNo = 0
    for sequence in sampleSeq:
        TriNucleotideNo = 0
        for TriNucleotide in map(operator.add, map(operator.add, sequence[::1], sequence[1::1]), sequence[2::1]):
            convertMatr[sampleNo][:, TriNucleotideNo] = PhyChemPropTable_Tri[TriNucleotide]
            TriNucleotideNo += 1
        sampleNo += 1
    
    return convertMatr

import torch
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, X_I, X_II, X_III, X_IV, Y):
        self.X_I_t = torch.Tensor(X_I)
        self.X_II_t = torch.Tensor(X_II)
        self.X_III_t = torch.Tensor(X_III)
        self.X_IV_t = torch.Tensor(X_IV)
        self.Y_t = torch.Tensor(Y)
        self.len = X_I.shape[0]

    def __getitem__(self, index):
        
        return self.X_I_t[index], self.X_II_t[index], self.X_III_t[index], self.X_IV_t[index], self.Y_t[index]
    
    def __len__(self):
       
        return self.len

class MyDataset_DNC(Dataset):
    def __init__(self, X_I, X_DNC, Y):
        self.X_I_t = torch.Tensor(X_I)
        self.X_DNC_t = torch.Tensor(X_DNC)
        self.Y_t = torch.Tensor(Y)
        self.len = X_I.shape[0]

    def __getitem__(self, index):
        
        return self.X_I_t[index], self.X_DNC_t[index], self.Y_t[index]
    
    def __len__(self):
       
        return self.len