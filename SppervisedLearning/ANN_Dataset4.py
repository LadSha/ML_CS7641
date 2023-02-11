#Reference https://github.com/bnsreenu/python_for_microscopists/blob/master/154_understanding_train_validation_loss_curves.py
#Reference https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-keras.md
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import seaborn as sns
from tensorflow import keras
from sklearn.model_selection import KFold
from DataPrep2 import get_data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from HelperFunctions import custom_f1, f1_m
from keras.callbacks import EarlyStopping, ModelCheckpoint
from numpy.random import seed
import tensorflow as tf

metric="f1_m"#custom_f1
metric="recall"
x_tr,y_tr, X_test, y_test = get_data()


X_train, X_val, y_train, y_val = train_test_split(x_tr, y_tr, test_size=0.15, random_state=0, stratify=y_tr)
print(len(y_train[y_train==0]))
print(len(y_train))

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
    recall= history.history["recall"]
    acc = history.history[f'{metric}']
    val_acc = history.history[f'val_{metric}']
    plt.plot(epochs, acc, 'y', label=f'Training {metric}')
    plt.plot(epochs, val_acc, 'r', label=f'Validation {metric}')
    plt.title(f'Training and validation {experiment_name}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{metric}')
    plt.legend()
    plt.show()




def complex_model():
    # Define the K-fold Cross Validator
    # num_folds = 5
    # verbosity=True
    # batch_size = 32
    # no_epochs = 50
    #
    # # Define per-fold score containers
    # acc_per_fold = []
    # loss_per_fold = []
    #
    # kfold = KFold(n_splits=num_folds, shuffle=True)
    #
    # # K-fold Cross Validation model evaluation
    # fold_no = 1
    # for train, test in kfold.split(x_train, y_train):
    #
    #     model = Sequential()
    #     model.add(Dense(16, input_dim=117, activation='relu'))
    #     model.add(Dense(16))
    #     model.add(Activation('relu'))
    #     model.add(Dense(1))
    #     model.add(Activation('sigmoid'))
    #     model.compile(loss='binary_crossentropy',
    #                   optimizer='adam',             #also try adam
    #                   metrics=[metric])
    #
    #     print(model.summary())
    #     # history = model.fit(X_train, y_train, verbose=1, epochs=500, batch_size=32,
    #     #                     validation_data=(X_test, y_test))
    #     history = model.fit(x_train[train], y_train[train],
    #                         batch_size=batch_size,
    #                         epochs=no_epochs,
    #                         verbose=verbosity,validation_split=.2)
    #
    #     # Generate generalization metrics
    #     scores = model.evaluate(x_train[test], y_train[test], verbose=0)
    #     print(
    #         f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
    #     acc_per_fold.append(scores[1] * 100)
    #     loss_per_fold.append(scores[0])
    #
    #     # Increase fold number
    #     fold_no = fold_no + 1
    #
    # plot_loss(history)
    # print()




    model = Sequential()
    model.add(Dense(16, input_dim=11, activation='relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    opt = keras.optimizers.Adam()
    model.compile(loss='binary_crossentropy',
                  optimizer=opt,             #also try adam
                  metrics=[metric])

    print(model.summary())
    history = model.fit(X_train, y_train, verbose=1, epochs=300, batch_size=512,
                        validation_data=(X_test, y_test))


    plot_loss(history, "complex_model")









def simple_model():
    seed(1)
    #simple model
#make sure to mention batch size, layers, nerons, learning rate , epochs
#
#     model = Sequential()
#     model.add(Dense(2,input_dim=X_train.shape[1], activation='relu'))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#     opt = keras.optimizers.Adam()#
#     model.compile(loss='binary_crossentropy',
#                   optimizer=opt,  # also try adam
#                   metrics=[ "Recall"])#f1_m
#     # class_weights = {0: .7, 1: 1}
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
#
#     history = model.fit(X_train, y_train, verbose=1, epochs=1000, batch_size=32,
#                         validation_data=(X_val, y_val),callbacks=[es])
#     #
#     _,f1_score= model.evaluate(X_test, y_test, verbose=0)
#     # print(f1_score)
#     l, acc = model.evaluate(X_test, y_test)
#     print(history.history)
#     # print("Accuracy = ", (acc * 100.0), "%")
#     plot_loss(history,"1layer-2node")
# #dropout
#     model = Sequential()
#     model.add(Dense(2, input_dim=X_train.shape[1], activation='relu'))
#     model.add(Dropout(0.05))
#     model.add(Dense(1))
#     model.add(Activation('sigmoid'))
#     opt = keras.optimizers.Adam(learning_rate=.001)  #
#     model.compile(loss='binary_crossentropy',
#                   optimizer=opt,  # also try adam
#                   metrics=["Recall"])  # f1_m
#     # class_weights = {0: .7, 1: 1}
#     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
#
#     history = model.fit(X_train, y_train, verbose=1, epochs=1000, batch_size=32,
#                         validation_data=(X_val, y_val), callbacks=[es])
#     #
#     _, f1_score = model.evaluate(X_test, y_test, verbose=0)
#     # print(f1_score)
#     l, acc = model.evaluate(X_test, y_test)
#     print(history.history)
#     # print("Accuracy = ", (acc * 100.0), "%")
#     plot_loss(history, "1layer-2node-.05dropout")


    # learning rate
    model = Sequential()
    nodes=5
    dropout=0.2
    lr=0.01
    model.add(Dense(nodes, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(dropout))
    # model.add(Dense(nodes, input_dim=X_train.shape[1], activation='relu'))



    model.add(Dense(1))
    model.add(Activation('LeakyReLU'))
    opt = keras.optimizers.Adam()  #
    model.compile(loss =tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=opt,  # also try adam
                  metrics=["Recall"])  # f1_m
    # class_weights = {0: .7, 1: 1}
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)

    history = model.fit(X_train, y_train, verbose=1, epochs=1000, batch_size=64,
                        validation_data=(X_val, y_val), callbacks=[es])
    #
    _, f1_score = model.evaluate(X_test, y_test, verbose=0)
    # print(f1_score)
    l, acc = model.evaluate(X_test, y_test)
    print(history.history)
    # print("Accuracy = ", (acc * 100.0), "%")
    plot_loss(history, f"1layer-{nodes}node-{dropout}dropout")

    # model = Sequential()
    # model.add(Dense(1,input_dim=X_train.shape[1], activation='relu'))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    # opt = keras.optimizers.Adam()#
    # model.compile(loss='binary_crossentropy',
    #               optimizer=opt,  # also try adam
    #               metrics=[ "Recall"])#f1_m
    # # class_weights = {0: .7, 1: 1}
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    #
    # history = model.fit(X_train, y_train, verbose=1, epochs=1000, batch_size=32,
    #                     validation_data=(X_val, y_val),callbacks=[es])
    # #
    # _,f1_score= model.evaluate(X_test, y_test, verbose=0)
    # # print(f1_score)
    # l, acc = model.evaluate(X_test, y_test)
    # print(history.history)
    # print("Accuracy = ", (acc * 100.0), "%")
    # plot_loss(history,"1layer-1node")



if __name__=="__main__":
    # complex_model()
    simple_model()
    # model = Sequential()
    # model.add(Dense(2, input_dim=X_train.shape[1], activation='relu'))
    # model.add(Dropout(0.05))
    # model.add(Dense(1))
    # model.add(Activation('sigmoid'))
    # opt = keras.optimizers.Adam(learning_rate=.001)  #
    # model.compile(loss='binary_crossentropy',
    #               optimizer=opt,  # also try adam
    #               metrics=["Recall"])  # f1_m
    # # class_weights = {0: .7, 1: 1}
    # es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=100)
    #
    # history = model.fit(X_train, y_train, verbose=1, epochs=1000, batch_size=32,
    #                     validation_data=(X_val, y_val), callbacks=[es])
    #
    # l, rec = model.evaluate(X_test, y_test)
    #
    # # print("Recall = ", (rec * 100.0), "%")
    # plot_loss(history, "1layer-2node-.05dropout")
    #
