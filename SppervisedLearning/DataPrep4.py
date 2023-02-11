import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler


def get_data():
    dataset = pd.read_csv(f'../Data/cancer.csv')
    #Reference https://towardsdatascience.com/building-a-simple-machine-learning-model-on-breast-cancer-data-eca4b3b99fa3
    X = dataset.iloc[:, 2:31].values
    Y = dataset.iloc[:, 1].values
    labelencoder_Y = LabelEncoder()
    Y = labelencoder_Y.fit_transform(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train,Y_train, X_test, Y_test
if __name__=="__main__":
    get_data()




