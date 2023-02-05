#Reference https://github.com/bnsreenu/python_for_microscopists/blob/master/154_understanding_train_validation_loss_curves.py

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import seaborn as sns
from tensorflow import keras

    #complex model
def complex_model():
    model = Sequential()
    model.add(Dense(16, input_dim=117, activation='relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',             #also try adam
                  metrics=['accuracy'])

    print(model.summary())
    history = model.fit(X_train, y_train, verbose=1, epochs=500, batch_size=32,
                        validation_data=(X_test, y_test))


    #simple model
#make sure to mention batch size, layers, nerons, learning rate , epochs

    model = Sequential()
    model.add(Dense(2,input_dim=117, activation='relu'))
    # # model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',  # also try adam
                  metrics=['accuracy'])

    print(model.summary())
    #
    history = model.fit(X_train, y_train, verbose=1, epochs=50, batch_size=32,
    #                     validation_data=(X_test, y_test))
    #
    _, acc = model.evaluate(X_test, y_test)
    print("Accuracy = ", (acc * 100.0), "%")




#plot the training and validation accuracy and loss at each epoch
    #simple model

    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'y', label='Training loss')
    # plt.plot(epochs, val_loss, 'r', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.show()
    #
    #
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    # plt.plot(epochs, acc, 'y', label='Training acc')
    # plt.plot(epochs, val_acc, 'r', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # plt.show()

    # Predicting the Test set results
    # y_pred = model.predict(X_test)
    # y_pred = (y_pred > 0.5)
    #
    # # Making the Confusion Matrix
    # from sklearn.metrics import confusion_matrix
    #
    # cm = confusion_matrix(y_test, y_pred)
    #
    # sns.heatmap(cm, annot=True)
    # plt.show()
