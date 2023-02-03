import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler


def get_data():
    dataset = pd.read_csv(f'../Data/mushrooms.csv')
    y_train = dataset['class']
    x_train = dataset.drop(labels=['class'], axis=1)

    le = LabelEncoder()

    cols = x_train.columns.values
    for col in cols:
        x_train[col] = le.fit_transform(x_train[col])

    y_train = le.fit_transform(y_train)

    ohe = OneHotEncoder()
    x_train = ohe.fit_transform(x_train).toarray()
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)

    return x_train,y_train

