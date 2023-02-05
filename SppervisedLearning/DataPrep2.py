import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler

def get_data():
    df = pd.read_csv(f'../Data/winequalityN.csv')
#Refernece https://www.geeksforgeeks.org/wine-quality-prediction-machine-learning/
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    df.replace({'white': 1, 'red': 0}, inplace=True)
    df = df.drop('total sulfur dioxide', axis=1)

    features = df.drop(['quality', 'best quality'], axis=1)
    target = df['best quality']

    xtrain, xtest, ytrain, ytest = train_test_split(
        features, target, test_size=0.2, random_state=40)

    norm = MinMaxScaler()
    xtrain = norm.fit_transform(xtrain)
    xtest = norm.transform(xtest)

    return xtrain,ytrain, xtest, ytest



