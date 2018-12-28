# -*- coding: utf-8 -*-
""" Author: Jared Williams

This is a online testable shell of the model, weights, and tokenizer
created by the sentiment_analysis_NN.py file.

"""
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
import pickle

MAX_NB_WORDS = 20000
MAX_SEQUENCE_LENGTH = 1000
EMBEDDING_DIM = 100


# imports the pre-trained neural network, weights, and tokenizer
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
pickle_off1 = open("tokenizer.pkl", "rb")
tokenizer = pickle.load(pickle_off1)


def main():
    """An infinite loop for testing purposes."""
    while True:
        review = input("Hey how was the movie?")
        if predict_sentiment(review)[0] < .5:
            print("That's a positive review")
            print(predict_sentiment(review)[0])
            print(predict_sentiment(review)[1])
        else:
            print("That's a negative review")


def predict_sentiment(input_string):
    """runs a review string through a trained TensorFlow model,
       using the word embeddings assigned by the fitted
       tokenizer, and returns a list of the decimal probabilities,
       negative first, then positive.

       Args:
           input_string: the review string.
        Return:
            the negative/positive probability list.
                """
    input_string = (input_string.encode('ascii', 'ignore').decode("utf-8"))
    input_string = re.sub(r"\\", "", input_string)
    input_string = re.sub(r"\'", "", input_string)
    input_string = re.sub(r"\"", "", input_string)
    input_string = input_string*50
    sequences = tokenizer.texts_to_sequences([input_string])
    sequences = np.asarray(sequences)
    data = np.reshape(sequences, (1, sequences.size))

    data = pad_sequences(data, maxlen=MAX_SEQUENCE_LENGTH)

    prediction = loaded_model.predict(data)
    return [prediction[0][0], prediction[0][1]]


if __name__ == '__main__':
    main()


