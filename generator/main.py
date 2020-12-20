import pandas as pd

START = '÷'
END = '■'


def main():
    raw_data = read_data()
    print("raw data:\n")
    print(raw_data)
    formatted_data = format_data(raw_data)
    print("formatted data:\n")
    print(formatted_data)


def read_data():
    dataset = pd.read_csv('../generator/dataset/dataset.csv', delimiter=',')
    data = dataset.drop(['publish_date'], axis=1).rename(columns={'headline_text': 'headline'})
    return data


def format_data(raw_dataset):
    return 0


if __name__ == "__main__":
    main()
