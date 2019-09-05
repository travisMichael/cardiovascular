import pandas as pd


def pre_process_loan_data(path):
    df = pd.read_csv(path + "cardio_train.csv", delimiter=";")


if __name__ == "__main__":
    pre_process_loan_data('../')
