import pandas as pd


def main():
    print("Hello World!")


def read_data():
    dataset = pd.read_csv('../generator/dataset/dataset.csv', delimiter=',')
    data = dataset.drop(['publish_date'], axis=1).rename(columns={'headline_text': 'headline'})
    return data


if __name__ == "__main__":
    main()
