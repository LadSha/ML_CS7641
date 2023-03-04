#Reference https://github.com/bnsreenu/python_for_microscopists/blob/master/154_understanding_train_validation_loss_curves.py
#Reference https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import KFold
from DataPrep1 import get_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from HelperFunctions import f1_m
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy.random import seed

metric="recall"#custom_f1
x_tr,y_tr, X_test, y_test = get_data()


X_train, X_val, y_train, y_val = train_test_split(x_tr, y_tr, test_size=0.15, random_state=0, stratify=y_tr)

#complex model

def plot_loss(history,experiment_name):

# plot the training and validation accuracy and loss at each epoch

    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(f'Training and validation loss {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
#
#
    acc = history.history[f'{metric}']
    val_acc = history.history[f'val_{metric}']
    plt.plot(epochs, acc, 'y', label=f'Training {metric}')
    plt.plot(epochs, val_acc, 'r', label=f'Validation {metric}')
    plt.title(f'Training and validation {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.show()







def experiment():
    seed(1)
    #simple model
#make sure to mention batch size, layers, nerons, learning rate , epochs
    nodes=4
    dropout="no "
    model = Sequential()
    model.add(Dense(nodes,input_dim=X_train.shape[1], activation='relu'))
    # model.add(Dropout(dropout))


    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    opt = keras.optimizers.Adam()#
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,  # also try adam
                  metrics=["Recall"])#f1_m
    # class_weights = {0: .7, 1: 1}
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

    history = model.fit(X_train, y_train, verbose=1, epochs=500, batch_size=64,
                        validation_data=(X_val, y_val),callbacks=[es])
    #
    _,f1_score= model.evaluate(X_test, y_test, verbose=0)
    # print(f1_score)

    # print("Accuracy = ", (acc * 100.0), "%")
    plot_loss(history,f"1layer-{nodes}node-.{dropout}dropout")


if __name__=="__main__":
    experiment()
#
