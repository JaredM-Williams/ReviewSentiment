# -*- coding: utf-8 -*-
""" Author: Jared Williams

This program creates and saves a model that is skilled at classifying
a string as having overall positive or negative sentiment.

"""
import os
import re
import pickle

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout
from keras.models import Model

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def clean_str(string):
    """Removes possible escape characters,
       from strings and sets them lowercase.

    Args:
        string: String that needs cleaning.
    Returns:
        The cleaned string.
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def train_model(model, x_val, y_val, x_train, y_train):
    """Trains the model, and saves the
       Tensorboard callback.

        Args:
            model: The untrained model.
            x_val: The validation data.
            y_val: The validation labels.
            x_train: The training data.
            y_train: The training labels.
        """
    tb_callback = keras.callbacks.TensorBoard(log_dir="./logs_temp",
                                              histogram_freq=0, write_graph=True, write_images=True)

    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=10, batch_size=128,
              callbacks=[tb_callback])


def save_neural_network(model):
    """Saves the trained model and its weights
       to a json and h3 file, respectively.

        Args:
            model: the trained model.
        """
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("model.h5")
    print("Saved model to disk")


def import_glove_embeddings():
    """Imports the gloVe embeddings as a
       dictionary, with the word as the key.

        Returns:
            The embedding dictionary.
        """
    GLOVE_DIR = "/Users/jaredwilliams/Documents/AI/NLP"
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, "glove.6B.100d.txt"))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def gather_needed_embeddings(word_index, embeddings_index):
    """Aligns the word's token index with the
       word's gloVe vector representation to
       create a matrix.

        Args:
            word_index: the list of relevant words.
            embeddings_index: the dictionary of gloVe embeddings.
        Returns:
            The embedding dictionary.
            """
    embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def main():
    """Trains the Neural Network and saves
       it in various files.
            """
    # imports the data
    data_train = pd.read_csv("labeledTrainData.tsv", sep="\t")
    print(data_train.shape)

    texts = []
    labels = []

    # cleans the text and separates the labels from the data
    for idx in range(data_train.review.shape[0]):
        text = BeautifulSoup(data_train.review[idx], features="html.parser")
        texts.append(clean_str(text.get_text().encode("ascii", "ignore").decode("utf-8")))
        labels.append(data_train.sentiment[idx])

    # creates a unique tokenizer and fits it to the data,
    # then transforms the data into its numerical tokens
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    with open("tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)

    # a list of the relevant words
    word_index = tokenizer.word_index
    print("Found %s unique tokens." % len(word_index))

    # pads the sentences to a consistent length
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # tells tf that the labels are binary
    labels = to_categorical(np.asarray(labels))
    print("Shape of data tensor:", data.shape)
    print("Shape of label tensor:", labels.shape)

    # shuffles the data and realigns the labels
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    # separates the training and validation data
    x_train = data[:-nb_validation_samples]
    y_train = labels[:-nb_validation_samples]
    x_val = data[-nb_validation_samples:]
    y_val = labels[-nb_validation_samples:]

    print("Positive/negative split in training and validation set ")
    print(y_train.sum(axis=0))
    print(y_val.sum(axis=0))

    embeddings_index = import_glove_embeddings()
    print("Total %s word vectors in Glove 6B 100d." % len(embeddings_index))

    embedding_matrix = gather_needed_embeddings(word_index, embeddings_index)

    # We construct a neural network with these specs:
    # An Embedding Layer
    # Three Convolution Layers, with Max Pools in between,
    # the last of which is significantly bigger.
    # two dense layers, ending in a Softmax binary activation
    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=True)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedded_sequences = embedding_layer(sequence_input)
    l_cov1 = Conv1D(128, 5, activation="relu")(embedded_sequences)
    l_pool1 = MaxPooling1D(5)(l_cov1)
    l_drop1 = Dropout(.2)(l_pool1)
    l_cov2 = Conv1D(128, 5, activation="relu")(l_drop1)
    l_pool2 = MaxPooling1D(5)(l_cov2)
    l_drop1 = Dropout(.2)(l_pool2)
    l_cov3 = Conv1D(128, 5, activation="relu")(l_drop1)
    l_pool3 = MaxPooling1D(35)(l_cov3)
    l_flat = Flatten()(l_pool3)
    l_dense = Dense(128, activation="relu")(l_flat)
    l_drop3 = Dropout(.4)(l_dense)
    preds = Dense(2, activation="softmax")(l_drop3)

    model = Model(sequence_input, preds)
    model.compile(loss="categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=["acc"])

    model.summary()

    train_model(model, x_val, y_val, x_train, y_train)

    save_neural_network(model)


if __name__ == "__main__":
    main()
