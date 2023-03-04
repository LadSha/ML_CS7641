import pandas as pd
from sklearn.preprocessing import OneHotEncoder,LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler

def get_data():
    df = pd.read_csv(f'../../Data/winequalityN.csv')
#Refernece https://www.geeksforgeeks.org/wine-quality-prediction-machine-learning/
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mean())
    df.replace({'M': 1, 'F': 0}, inplace=True)
    df['old'] = [1 if x > 15 else 0 for x in df.Rings]
    poor_qual=df[df.old==0]
    # df=pd.concat([df,poor_qual])


    features = df.drop(['Rings'], axis=1)
    target = df['old']
    xtrain, xtest, ytrain, ytest = train_test_split(
        features, target, test_size=0.25, random_state=42)

    norm = MinMaxScaler()
    xtrain = norm.fit_transform(xtrain)
    xtest = norm.transform(xtest)

    return xtrain,ytrain, xtest, ytest
if __name__=="__main__":
    xtrain, ytrain, xtest, ytest=get_data()




