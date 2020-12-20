from dataclasses import dataclass

import numpy as np
import pandas as pd
from keras.layers import Dense, Bidirectional, Embedding, LSTM
from keras.layers import SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

START = '÷'
END = '■'


@dataclass
class Parameters:
    max_features: int
    max_len: int
    epochs: int
    start: str
    end: str


def get_parameters():
    return Parameters(
        max_features=5,
        max_len=15,
        epochs=3,
        start=START,
        end=END
    )


def kek():
    lol = 5


def build_model(epochs, max_features, X, Y):
    model = Sequential()

    # Embedding and GRU
    model.add(Embedding(max_features, 300))
    model.add(SpatialDropout1D(0.33))
    model.add(Bidirectional(LSTM(30)))

    # Output layer
    model.add(Dense(max_features, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, Y, epochs=epochs, batch_size=128, verbose=1)

    model.save_weights('model{}.h5'.format(epochs))
    model.evaluate(X, Y)
    return model


def main():
    raw_data = read_data()
    print("raw data:\n")
    print(raw_data)

    # formatted_data = format_data(raw_data, max_features, max_len)
    # print("formatted data:\n")
    # print(formatted_data)


def create_idx_to_words(tokenizer):
    """When sampling from the distribution, we do not know which word is being
    sampled, only its index. We need a way to go from index to word. Unfortunately,
    the tokenizer class only contains a dictionary of {word: index} items. We will
    reverse that dictionary to get {index: word} items. That way, going from
    indices to words is much faster."""
    idx_to_words = {value: key for key, value in tokenizer.word_index.items()}
    return idx_to_words


def read_data():
    dataset = pd.read_csv('../generator/dataset/dataset.csv', delimiter=',')
    data = dataset.drop(['publish_date'], axis=1).rename(columns={'headline_text': 'headline'})
    return data


def format_data(data, max_features, maxlen, shuffle=False):
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)

        # Add start and end tokens
    data['headline'] = START + ' ' + data['headline'].str.lower() + ' ' + END

    text = data['headline']

    # Tokenize text
    filters = "!\"#$%&()*+,-./;<=>?@[\\]^_`{|}~\t\n"
    tokenizer = Tokenizer(num_words=max_features, filters=filters)
    tokenizer.fit_on_texts(list(text))
    corpus = tokenizer.texts_to_sequences(text)

    # Build training sequences of (context, next_word) pairs.
    # Note that context sequences have variable length. An alternative
    # to this approach is to parse the training data into n-grams.
    X, Y = [], []
    for line in corpus:
        for i in range(1, len(line) - 1):
            X.append(line[:i + 1])
            Y.append(line[i + 1])

    # Pad X and convert Y to categorical (Y consisted of integers)
    X = pad_sequences(X, maxlen=maxlen)
    Y = to_categorical(Y, num_classes=max_features)

    return X, Y, tokenizer


def sample(preds, temp=1.0):
    """
    Sample next word given softmax probabilities, using temperature.

    Taken and modified from:
    https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py
    """
    EPSILON = 10e-16
    preds = (np.asarray(preds)).astype('float64')
    preds = np.log(preds) / temp
    preds = np.exp(preds) / np.sum(np.exp(preds))
    probs = np.random.multinomial(1, preds, 1)
    return np.argmax(probs)


# def sample(softmax, temp=1.0):
#     EPSILON = 10e-16 # to avoid taking the log of zero
#     #print(preds)
#     (np.array(softmax) + EPSILON).astype('float64')
#     preds = np.log(softmax) / temp
#     #print(preds)
#     exp_preds = np.exp(preds)
#     #print(exp_preds)
#     preds = exp_preds / np.sum(exp_preds)
#     #print(preds)
#     probas = np.random.multinomial(1, preds, 1)
#     return probas[0]


def process_input(text, tokenizer, max_len):
    """Tokenize and pad input text"""
    tokenized_input = tokenizer.texts_to_sequences([text])[0]
    return pad_sequences([tokenized_input], maxlen=max_len - 1)


def generate_text(input_text, model, idx_to_words, tokenizer, max_len, n=7, temp=1.0):
    """Takes some input text and feeds it to the model (after processing it).
    Then, samples a next word and feeds it back into the model until the end
    token is produced.

    :input_text: A string or list of strings to be used as a generation seed.
    :model:      The model to be used during generation.
    :temp:       A float that adjusts how 'volatile' predictions are. A higher
                 value increases the chance of low-probability predictions to
                 be picked."""
    if type(input_text) is str:
        sent = input_text
    else:
        sent = ' '.join(input_text)

    tokenized_input = process_input(input_text, tokenizer, max_len)

    while True:
        preds = model.predict(tokenized_input, verbose=0)[0]
        pred_idx = sample(preds, temp=temp)
        pred_word = idx_to_words[pred_idx]

        if pred_word == END:
            return sent

        sent += ' ' + pred_word
        #         print(sent)
        #         tokenized_input = process_input(sent[-n:])
        tokenized_input = process_input(sent, tokenizer, max_len)


if __name__ == "__main__":
    main()
