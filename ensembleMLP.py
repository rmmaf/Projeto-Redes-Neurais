from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier
from scipy.stats import ks_2samp as ksTest
from sklearn.utils import shuffle
from sklearn import metrics
import numpy as np
import pandas as pd

tr = pd.read_hdf("datasets/repeat/Train.h5", key='train')
va = pd.read_hdf("datasets/repeat/Validation.h5", key='validation')
te = pd.read_hdf("datasets/repeat/Test.h5", key='test')

tr = tr.append(va)
tr = shuffle(tr)
tr1 = tr.iloc[:,:-1]
tr2 = tr['IND_BOM_1_1']


def mlp1():
        clf = MLPClassifier(
        hidden_layer_sizes=(1000,),
        solver='sgd',
        activation='relu',
        learning_rate='constant',
        learning_rate_init=0.03,
        early_stopping=True,
        validation_fraction=0.1)
        clf.fit(tr1, tr2)
        #rClass = clf.predict(te.iloc[:,:-1])
        #rProba = clf.predict_proba(te.iloc[:,:-1])[:,1]
        return clf


def mlp2():
        clf = MLPClassifier(
        hidden_layer_sizes=(200,),
        solver='sgd',
        activation='relu',
        learning_rate='constant',
        learning_rate_init=0.03,
        early_stopping=True,
        validation_fraction=0.1)
        clf.n_layers_ = 2
        clf.fit(tr1, tr2)
        #rClass = clf.predict(te.iloc[:,:-1])
        #rProba = clf.predict_proba(te.iloc[:,:-1])[:,1]
        return clf

def mlp3(): 
    clf = MLPClassifier(
    hidden_layer_sizes=(200,),
    solver='sgd',
    activation='relu',
    learning_rate='constant',
    learning_rate_init=0.03,
    early_stopping=True,
    validation_fraction=0.1)
    clf.n_layers_ = 3
    clf.fit(tr1, tr2)
    #rClass = clf.predict(te.iloc[:,:-1])
    #rProba = clf.predict_proba(te.iloc[:,:-1])[:,1]
    return clf


def med(rProba, rClass): 
    print('MSE:', metrics.mean_squared_error(te['IND_BOM_1_1'], rProba))
    print('KS Test:', ksTest(te['IND_BOM_1_1'], rProba)[0])
    print('ROC AUC:', metrics.roc_auc_score(te['IND_BOM_1_1'], rProba))
    print('Accuracy:', metrics.accuracy_score(te['IND_BOM_1_1'], rClass))
    print('Precision, Recall and FScore:')
    print(metrics.precision_recall_fscore_support(te['IND_BOM_1_1'], rClass, average='binary')[:-1])
    print('Confusion Matrix:')
    print(metrics.confusion_matrix(te['IND_BOM_1_1'], rClass))

clf1 = mlp1()
rClass = clf1.predict(te.iloc[:,:-1])
rProba = clf1.predict_proba(te.iloc[:,:-1])[:,1]
med(rProba,rClass)

clf2 = mlp2()
rClass = clf2.predict(te.iloc[:,:-1])
rProba = clf2.predict_proba(te.iloc[:,:-1])[:,1]
med(rProba,rClass)

clf3 = mlp3()
rClass = clf3.predict(te.iloc[:,:-1])
rProba = clf3.predict_proba(te.iloc[:,:-1])[:,1]
med(rProba,rClass)

eclf = VotingClassifier(estimators=[('mlp1', clf1),('mlp2', clf2),('mlp3', clf3)], voting='soft')
eclf = eclf.fit(tr1,tr2)
rClass = eclf.predict(te.iloc[:,:-1])
rProba = eclf.predict_proba(te.iloc[:,:-1])[:,1]
med(rProba,rClass)

print('MSE:', metrics.mean_squared_error(te['IND_BOM_1_1'], rProba))
print('KS Test:', ksTest(te['IND_BOM_1_1'], rProba)[0])
print('ROC AUC:', metrics.roc_auc_score(te['IND_BOM_1_1'], rProba))
print('Accuracy:', metrics.accuracy_score(te['IND_BOM_1_1'], rClass))
print('Precision, Recall and FScore:')
print(metrics.precision_recall_fscore_support(te['IND_BOM_1_1'], rClass, average='binary')[:-1])
print('Confusion Matrix:')
print(metrics.confusion_matrix(te['IND_BOM_1_1'], rClass))