import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim

MAXPOOL1D_KERNEL_SIZE = 2

CONV1D_KERNEL_SIZE = 3
CONV1D_FEATURE_SIZE_BLOCK1 = 32
CONV1D_FEATURE_SIZE_BLOCK2 = 16
CONV1D_FEATURE_SIZE_BLOCK3 = 8

FULLY_CONNECTED_LAYER1_SIZE = 64
FULLY_CONNECTED_LAYER2_SIZE = 32

BN1D_FEATURE_LAYER_SIZE = 128

### OneHotMonoMer Network ###
class CNNNet_OneHotMonoMer(nn.Module):
    def __init__(self, dropout_prob):
        super(CNNNet_OneHotMonoMer, self).__init__()
        self.dropout_prob = dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv1d(4, CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool2 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_FEATURE_SIZE_BLOCK3, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK3),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool3 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        output = self.pool1(self.conv1(inputs))
        output = self.pool2(self.conv2(output))
        output = self.pool3(self.conv3(output))
        output = output.view(batch_size, -1) # 72
        
        return output

### OneHotDiMer Network ###
class CNNNet_OneHotDiMer(nn.Module):
    def __init__(self, dropout_prob):
        super(CNNNet_OneHotDiMer, self).__init__()
        self.dropout_prob = dropout_prob
        self.conv1 = nn.Sequential( 
            nn.Conv1d(16, CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_KERNEL_SIZE), 
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool2 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_FEATURE_SIZE_BLOCK3, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK3),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool3 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)
        output = self.pool1(self.conv1(inputs))
        output = self.pool2(self.conv2(output))
        output = self.pool3(self.conv3(output))
        output = output.view(batch_size, -1)

        return output

### OneHotTriMer Network ###
class CNNNet_OneHotTriMer(nn.Module):
    def __init__(self, dropout_prob):
        super(CNNNet_OneHotTriMer, self).__init__()
        self.dropout_prob = dropout_prob
        self.conv1 = nn.Sequential( 
            nn.Conv1d(64, CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_KERNEL_SIZE), 
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool2 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_FEATURE_SIZE_BLOCK3, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK3),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool3 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)
        output = self.pool1(self.conv1(inputs))
        output = self.pool2(self.conv2(output))
        output = self.pool3(self.conv3(output))
        output = output.view(batch_size, -1) # 72

        return output

### PhyChemDi Network ###
class CNNNet_PhyChemDi(nn.Module):
    def __init__(self, dropout_prob):
        super(CNNNet_PhyChemDi, self).__init__()
        self.dropout_prob = dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv1d(12, CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool2 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_FEATURE_SIZE_BLOCK3, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK3),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool3 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
        
    def forward(self, inputs):
        batch_size = inputs.size(0)
        output = self.pool1(self.conv1(inputs))
        output = self.pool2(self.conv2(output))
        output = self.pool3(self.conv3(output))
        output = output.view(batch_size, -1) # 72
        
        return output

### PhyChemTri Network ###
class CNNNet_PhyChemTri(nn.Module):
    def __init__(self, dropout_prob):
        super(CNNNet_PhyChemTri, self).__init__()
        self.dropout_prob = dropout_prob
        self.conv1 = nn.Sequential(
            nn.Conv1d(12, CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK1),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool1 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK1, CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK2),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool2 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(CONV1D_FEATURE_SIZE_BLOCK2, CONV1D_FEATURE_SIZE_BLOCK3, CONV1D_KERNEL_SIZE),
            nn.BatchNorm1d(CONV1D_FEATURE_SIZE_BLOCK3),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.pool3 = nn.Sequential(
            nn.MaxPool1d(MAXPOOL1D_KERNEL_SIZE)
        )
    
    def forward(self, inputs):
        batch_size = inputs.size(0)
        output = self.pool1(self.conv1(inputs))
        output = self.pool2(self.conv2(output))
        output = self.pool3(self.conv3(output))
        output = output.view(batch_size, -1) # 72
        
        return output

### Branched Convolution Neural Network ###
class BranchedCNN_Net(nn.Module):
    def __init__(self, dropout_prob=0.28):
        super(BranchedCNN_Net, self).__init__()
        self.dropout_prob = dropout_prob
        self.branch1 = CNNNet_OneHotMonoMer(self.dropout_prob)
        self.branch2 = CNNNet_OneHotTriMer(self.dropout_prob)
        self.branch3 = CNNNet_PhyChemDi(self.dropout_prob)
        self.branch4 =CNNNet_PhyChemTri(self.dropout_prob)
        self.bn1 = nn.BatchNorm1d(216)
        self.fc1 = nn.Sequential(
            nn.Linear(216, FULLY_CONNECTED_LAYER1_SIZE),
            nn.BatchNorm1d(FULLY_CONNECTED_LAYER1_SIZE),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(FULLY_CONNECTED_LAYER1_SIZE, FULLY_CONNECTED_LAYER2_SIZE),
            nn.BatchNorm1d(FULLY_CONNECTED_LAYER2_SIZE),
            nn.ReLU(),
            nn.Dropout(self.dropout_prob)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(216, 3)
        )
    
    def forward(self, inputs_I, inputs_II, inputs_III, inputs_IV):
        feature_I = self.branch1(inputs_I)
        feature_II = self.branch2(inputs_II)
        feature_III = self.branch3(inputs_III)
        feature_IV = self.branch4(inputs_IV)
        features = torch.cat((feature_I, feature_II, feature_III), 1)
        features = self.bn1(features)
        output = self.fc3(features)
        output = F.softmax(output, dim=1)

        return output

### iPSW(PseDNC-DL) Network ###
class iPSW_PseDNC_DL_Net(nn.Module):
    def __init__(self, dropout_prob):
        super(iPSW_PseDNC_DL_Net, self).__init__()
        self.dropout_prob = dropout_prob
        self.branch1 = CNNNet_OneHotMonoMer(self.dropout_prob)
        self.bn1 = nn.BatchNorm1d(88)
        self.fc1 = nn.Sequential(
            nn.Linear(88, 3)
        )
    
    def forward(self, inputs_I, inputs_DNC):
        batch_size = inputs_DNC.size(0)
        feature_I = self.branch1(inputs_I)
        feature_DNC = inputs_DNC.view(batch_size, -1)
        features = torch.cat((feature_I, feature_DNC), 1)
        features = self.bn1(features)
        output = self.fc1(features)
        output = F.softmax(output, dim=1)

        return output