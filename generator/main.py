from dataclasses import dataclass

import nltk
import numpy as np
import pandas as pd
from keras.layers import Dense, Bidirectional, Embedding, LSTM
from keras.layers import SpatialDropout1D
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


@dataclass
class Parameters:
    model_name: str
    nrows: int
    max_features: int
    max_len: int
    epochs: int
    start: str
    end: str


def get_parameters():
    return Parameters(
        model_name='model',
        nrows=200000,
        max_features=3500,
        max_len=20,
        epochs=3,
        start='÷',
        end='■'
    )

def create_model(max_features):
    model = Sequential()

    # Embedding and GRU
    model.add(Embedding(max_features, 300))
    model.add(SpatialDropout1D(0.33))
    model.add(Bidirectional(LSTM(30)))

    # Output layer
    model.add(Dense(max_features, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def build_model(name, epochs, max_features, X, Y):
    model = create_model(max_features)

    model.fit(X, Y, epochs=epochs, batch_size=128, verbose=1, validation_split=0.2)

    model.save(f'{name}.h5')
    return model



def read_model(name):
    model = load_model(f'{name}.h5')
    return model


def read_model_weights(name, model):
    model.load_weights(f'{name}.h5')


def main():
    ref = ['lol', 'kek', 'cheburek', 'z', 'z,', 'z']
    hyp1 = ['lol', 'lul', 'cheburek', 'z', 'z,', 'z']
    hyp2 = ['lol', 'kuk', 'cheburek', 'z', 'z,', 'z']
    hyp3 = ['lol', 'kuk', 'cheburek', 'z', 'z,', 'z']

    bleu1 = nltk.translate.bleu_score.sentence_bleu([ref], hyp1)
    bleu2 = nltk.translate.bleu_score.sentence_bleu([ref], hyp2)
    bleu3 = nltk.translate.bleu_score.sentence_bleu([ref], hyp3)
    bleu4 = nltk.translate.bleu_score.sentence_bleu([ref, hyp3], hyp3)

    print("bleu1:", bleu1, '\n')
    print("bleu2:", bleu2, '\n')
    print("bleu3:", bleu3, '\n')
    print("bleu4:", bleu4, '\n')


def create_idx_to_words(tokenizer):
    """When sampling from the distribution, we do not know which word is being
    sampled, only its index. We need a way to go from index to word. Unfortunately,
    the tokenizer class only contains a dictionary of {word: index} items. We will
    reverse that dictionary to get {index: word} items. That way, going from
    indices to words is much faster."""
    idx_to_words = {value: key for key, value in tokenizer.word_index.items()}
    return idx_to_words


def read_data(nrows):
    dataset = pd.read_csv('../generator/dataset/dataset.csv', delimiter=',', nrows=nrows)
    data = dataset.drop(['publish_date'], axis=1).rename(columns={'headline_text': 'headline'})
    return data


def format_data(raw_data, max_features, maxlen, start, end, shuffle=False,):
    data = raw_data.copy(deep=True)
    if shuffle:
        data = data.sample(frac=1).reset_index(drop=True)

        # Add start and end tokens
    data['headline'] = start + ' ' + data['headline'].str.lower() + ' ' + end

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

    return X, Y, tokenizer, text


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


def process_input(text, tokenizer, max_len):
    """Tokenize and pad input text"""
    tokenized_input = tokenizer.texts_to_sequences([text])[0]
    return pad_sequences([tokenized_input], maxlen=max_len - 1)


def generate_text(input_text, model, idx_to_words, tokenizer, max_len, end, n=7, temp=1.0,):
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
        print("pred_idx: ", pred_idx, "pred_word: ", pred_word)

        if pred_word == end:
            print("return sent: ", sent)
            return sent

        sent += ' ' + pred_word
        tokenized_input = process_input(sent, tokenizer, max_len)


if __name__ == "__main__":
    main()
