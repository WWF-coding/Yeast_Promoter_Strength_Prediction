import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

def compareSVM(Seq_transform, Activity):

    X = Seq_transform
    y = Activity

    C_values = np.logspace(0, 10, 6, base=2)
    gamma_values = np.logspace(-4, -1, 4)
    param_grid = [{'C': C_values, 'gamma': gamma_values}]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    clf_SV = SVC(decision_function_shape='ovo', probability=True)
    grid_SVM = GridSearchCV(clf_SV, param_grid, cv=cv, scoring='roc_auc_ovo')
    grid_SVM.fit(X, y)     
    
    return grid_SVM

def compareRF(Seq_transform, Activity):
    
    X = Seq_transform
    y = Activity

    n_estimators = np.arange(20, 200, 20)
    param_grid = [{'n_estimators': n_estimators}]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    clf_RF = RandomForestClassifier()
    grid_RF = GridSearchCV(clf_RF, param_grid, cv=cv, scoring='roc_auc_ovo')
    grid_RF.fit(X, y)
    
    return grid_RF

def compareGBDT(Seq_transform, Activity):
    
    X = Seq_transform
    y = Activity

    learning_rate = np.logspace(-3, -1, 3)
    n_estimators = np.arange(20, 200, 20)
    param_grid = [{'learning_rate': learning_rate, 'n_estimators': n_estimators}]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    clf_GBDT = GradientBoostingClassifier()
    grid_GBDT = GridSearchCV(clf_GBDT, param_grid, cv=cv, scoring='roc_auc_ovo')
    grid_GBDT.fit(X, y)
    
    return grid_GBDT
