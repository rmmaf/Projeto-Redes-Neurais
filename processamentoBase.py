from sklearn.neighbors import KDTree
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import seaborn as sns
import random as rnd
import pandas as pd

ds = pd.read_csv("datasets/original/TRN", sep='\t')
ds.describe()

ds.drop(['INDEX', 'IND_BOM_1_2'], axis=1, inplace=True)
ds[:5]

values = {}
for column in ds.columns:
    values[column] = sorted(set(ds[column].values))

ds0 = ds[ds['IND_BOM_1_1'] <  0.5]
ds1 = ds[ds['IND_BOM_1_1'] >= 0.5]

sns.countplot('IND_BOM_1_1', data=ds)

def splitter(df):
    df = shuffle(df)
    
    r1 = int(df.shape[0] * 0.25)
    r2 = int(df.shape[0] * 0.25) + r1
    
    return df[:r1], df[r1:r2], df[r2:]

te0, va0, tr0 = splitter(ds0)
te1, va1, tr1 = splitter(ds1)

def repeat(df1, df2):
    size = df1.shape[0] - df2.shape[0]
    if size < 0:
        rand = np.random.randint(df1.shape[0], size=abs(size))
        df1 = df1.iloc[list(range(df1.shape[0])) + list(rand),:]
    elif size > 0:
        rand = np.random.randint(df2.shape[0], size=size)
        df2 = df2.iloc[list(range(df2.shape[0])) + list(rand),:]
    return df1, df2

SMOTE = False
tr0, tr1 = repeat(tr0, tr1)
va0, va1 = repeat(va0, va1)

def get_near(column, v):
    aux = (len(values[column]) - 1)*v
    return values[column][int(round(aux, 0))]

def calc(x, y):
    row = []
    for column in x.index:
        p = rnd.random()
        v = x[column] + (y[column] - x[column])*p
        v = get_near(column, v)
        row.append(v)
    return row

def smote(df1, df2, k):
    aux = []
    size = df1.shape[0] - df2.shape[0]
    if size > 0:
        df1, df2 = df2, df1
    kdt = KDTree(df1)
    while(len(aux) != abs(size)):
        num1 = rnd.randrange(df1.shape[0])
        row1 = df1.iloc[num1]
        nAux = rnd.randrange(2, k)
        num2 = kdt.query([row1], k=nAux, return_distance=False)[0][-1]
        row2 = df1.iloc[num2]
        row  = calc(row1, row2)
        aux.append(row)
    if len(aux) != 0:
        df1 = df1.append(pd.DataFrame(aux, columns=df1.columns), ignore_index=True)
    if size > 0:
        df1, df2 = df2, df1
    return df1, df2

print("Initial values:")
print("\tValidation: " + str(va0.shape[0]) + "/" + str(va1.shape[0]))
print("\tTrain: " + str(tr0.shape[0]) + "/" + str(tr1.shape[0]))
SMOTE = True
va0, va1 = smote(va0, va1, 3)
tr0, tr1 = smote(tr0, tr1, 3)
print("Final values (After SMOTE):")
print("\tValidation: " + str(va0.shape[0]) + "/" + str(va1.shape[0]))
print("\tTrain: " + str(tr0.shape[0]) + "/" + str(tr1.shape[0]))

tr = tr0.append(tr1, ignore_index=True)
va = va0.append(va1, ignore_index=True)
te = te0.append(te1, ignore_index=True)

tr = shuffle(tr)
va = shuffle(va)
te = shuffle(te)

tr['IND_BOM_1_1'].hist().plot()
plt.show()
va['IND_BOM_1_1'].hist().plot()
plt.show()
te['IND_BOM_1_1'].hist().plot()
plt.show()

if SMOTE:
    tr.to_hdf("datasets/smote/Train.h5", key='train')
    va.to_hdf("datasets/smote/Validation.h5", key='validation')
    te.to_hdf("datasets/smote/Test.h5", key='test')
else:
    tr.to_hdf("datasets/repeat/Train.h5", key='train')
    va.to_hdf("datasets/repeat/Validation.h5", key='validation')
    te.to_hdf("datasets/repeat/Test.h5", key='test')