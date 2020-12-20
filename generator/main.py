from dataclasses import dataclass

import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

START = '÷'
END = '■'


@dataclass
class Parameters:
    max_features: int
    max_len: int


def get_parameters():
    return Parameters(
        max_features=3500,
        max_len=200
    )


def main():


    raw_data = read_data()
    print("raw data:\n")
    print(raw_data)

    # formatted_data = format_data(raw_data, max_features, max_len)
    # print("formatted data:\n")
    # print(formatted_data)


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


if __name__ == "__main__":
    main()
