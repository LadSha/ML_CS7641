# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def train_model():
    # Use a breakpoint in the code line below to debug your script.
    dataset = pd.read_csv('../Data/mushrooms.csv')
    print((dataset[dataset["class"]=="e"]).shape)
    print((dataset[dataset["class"] == "p"]).shape)
    y_train = dataset['class']
    x_train = dataset.drop(labels =['class'],axis=1)

    le = LabelEncoder()

    cols = x_train.columns.values
    for col in cols:
        x_train[col] = le.fit_transform(x_train[col])

    y_train = le.fit_transform(y_train)

    ohe = OneHotEncoder()
    x_train = ohe.fit_transform(x_train).toarray()
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)


    x_train, x_test, y_train, y_test = train_test_split(x_train,y_train, test_size = 0.30, random_state = 42)
    x_valid, x_test, y_valid, y_test = train_test_split(x_test,y_test, test_size = 0.50, random_state = 42)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_model()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
