from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ks_2samp as ksTest
from sklearn.utils import shuffle
from sklearn import metrics
import pandas as pd

tr = pd.read_hdf("datasets/repeat/Train.h5", key='train')
va = pd.read_hdf("datasets/repeat/Validation.h5", key='validation')
te = pd.read_hdf("datasets/repeat/Test.h5", key='test')

tr = tr.append(va)
tr = shuffle(tr)

clf = RandomForestClassifier(n_estimators=200)
clf.fit(tr.iloc[:,:-1], tr['IND_BOM_1_1'])
rClass = clf.predict(te.iloc[:,:-1])
rProba = clf.predict_proba(te.iloc[:,:-1])[:,1]

print('MSE:', metrics.mean_squared_error(te['IND_BOM_1_1'], rProba))
print('KS Test:', ksTest(te['IND_BOM_1_1'], rProba)[0])
print('ROC AUC:', metrics.roc_auc_score(te['IND_BOM_1_1'], rProba))
print('Accuracy:', metrics.accuracy_score(te['IND_BOM_1_1'], rClass))
print('Precision, Recall and FScore:')
print(metrics.precision_recall_fscore_support(te['IND_BOM_1_1'], rClass, average='binary')[:-1])
print('Confusion Matrix:')
print(metrics.confusion_matrix(te['IND_BOM_1_1'], rClass))