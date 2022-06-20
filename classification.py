import time
from shutil import copyfile
import keras
import numpy as np
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, MaxPool2D, Dropout, GlobalAveragePooling2D, Rescaling
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import os
import tensorflow as tf

dirs = ["./models", "./wrong_predicted", "./divided_data"]

for i in dirs:
    dir = os.path.join(i)
    if not os.path.exists(dir):
        os.mkdir(dir)


def define_model_first():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(1, activation='softmax'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def define_model_one_block():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def define_model_two_block():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def define_model_three_block():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model


def define_model_own():
    model = Sequential()
    model.add(Rescaling(1. / 255))
    model.add(Conv2D(16, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation="softmax"))
    model.compile(optimizer="Adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model


def define_model_global_pooling():
    model = Sequential()
    model.add(GlobalAveragePooling2D(input_shape=(224, 224, 3)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(4, activation="softmax"))
    model.compile(optimizer="Adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model


def define_model_best():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (2, 2), padding="same", activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer="Adam",
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])
    return model


def run_test_harness(model_function, saveOrLoad, fileName, copyWrongPrediction=False):
    model = model_function()
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_it = datagen.flow_from_directory('./divided_data/train/',
                                           class_mode='binary', batch_size=64, target_size=(224, 224))
    test_it = datagen.flow_from_directory('./divided_data/test/',
                                          class_mode='binary', batch_size=64, target_size=(224, 224))

    print(fileName + ":")
    if saveOrLoad == "save":

        start = time.time()
        history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                      validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)
        end = time.time()
        print("Train time:")
        print(end - start)
        model.save("./models/" + fileName + ".h5")
    elif saveOrLoad == "load":
        model = keras.models.load_model("./models/" + fileName + ".h5")

    img = tf.keras.utils.load_img("./divided_data/test/mild/mild_9.jpg", target_size=(224, 224, 3))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.array([img])

    start = time.time()
    model.predict(img, verbose=0)
    end = time.time()
    print("Evaluation time of one image:")
    print(end - start)

    start = time.time()
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose=0)
    end = time.time()
    print("Evaluation time:")
    print(end - start)

    pred = model.predict(test_it)
    print("Accuracy:")
    print('> %.3f' % (acc * 100.0))
    y_pred = np.argmax(pred, axis=1)
    err_matrix = confusion_matrix(test_it.classes, y_pred)
    print("Error matrix:")
    print(err_matrix)

    if copyWrongPrediction:
        fnames = test_it.filenames
        for i, j, k in zip(pred, test_it.classes, fnames):
            if round(i[0]) != j:
                copyfile("./divided_data/test/" + str(k), "./wrong_predicted/" + fileName + "/" + str(k))


# Pierwszy model
run_test_harness(define_model_first, "save", "firstModel")

# Modele z laboratorium
run_test_harness(define_model_one_block, "save", "oneBlockModel")
run_test_harness(define_model_two_block, "save", "twoBlockModel")
run_test_harness(define_model_three_block, "save", "threeBlockModel")

# Model z własnych prób i błędów
run_test_harness(define_model_own, "save", "ownModel")

# Model z globalnym poolingiem
run_test_harness(define_model_global_pooling, "save", "globalPoolingModel")

# Najlepszy model
run_test_harness(define_model_best, "save", "bestModel")
