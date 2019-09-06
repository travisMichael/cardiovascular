import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, RobustScaler
# import sklearn.metrics as mt
# from sklearn.model_selection import cross_val_score

def pre_process_loan_data(path):
    df = pd.read_csv(path + "loan.csv", delimiter=",")
    print(df.shape)

    good_loan = len(df[(df.loan_status == 'Fully Paid') |
                       (df.loan_status == 'Current') |
                       (df.loan_status == 'Does not meet the credit policy. Status:Fully Paid')])
    print('Good/Bad Loan Ratio: %.2f%%' % (good_loan/len(df)*100))

    df['good_loan'] = np.where((df.loan_status == 'Fully Paid') |
                               (df.loan_status == 'Current') |
                               (df.loan_status == 'Does not meet the credit policy. Status:Fully Paid'), 1, 0)

    lack_of_data_idx = [x for x in df.count() < 887379*0.25]
    df.drop(df.columns[lack_of_data_idx], 1, inplace=True)

    df.drop(['emp_title', 'title',  'zip_code','verification_status','home_ownership', 'addr_state', 'hardship_flag',
             'debt_settlement_flag', 'disbursement_method',
            'issue_d', 'earliest_cr_line', 'last_pymnt_d','next_pymnt_d','last_credit_pull_d',], axis=1, inplace=True)

    df.dropna(inplace=True)

    columns = ['term', 'grade', 'sub_grade', 'emp_length', 'purpose', 'application_type',
               'pymnt_plan', 'initial_list_status']

    for col in columns:
        tmp_df = pd.get_dummies(df[col], prefix=col)
        df = pd.concat((df, tmp_df), axis=1)

    df.drop(['loan_status',
             'term',
             'grade',
             'sub_grade',
             'emp_length',
             'initial_list_status',
             'pymnt_plan',
             'purpose',
             'application_type'], axis=1, inplace=True)

    df = df.rename(columns= {'emp_length_< 1 year': 'emp_length_lt_1 year',
                             'emp_length_n/a': 'emp_length_na'})

    y = df['good_loan']
    X = df.ix[:, df.columns != 'good_loan']

    X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, random_state=44)

    rob_scaler = RobustScaler()

    X_train_R = rob_scaler.fit_transform(X_train)
    X_test_R = rob_scaler.transform(X_test)

    if not os.path.exists(path + 'data/loan/train'):
        os.makedirs(path + 'data/loan/train')
    if not os.path.exists(path + 'data/loan/test'):
        os.makedirs(path + 'data/loan/test')

    x_train_file = open(path + 'data/loan/train/x', 'wb')
    y_train_file = open(path + 'data/loan/train/y', 'wb')
    x_test_file = open(path + 'data/loan/test/x', 'wb')
    y_test_file = open(path + 'data/loan/test/y', 'wb')

    pickle.dump(X_train_R, x_train_file)
    pickle.dump(y_train, y_train_file)
    pickle.dump(X_test_R, x_test_file)
    pickle.dump(y_test, y_test_file)

    x_train_file.close()
    y_train_file.close()
    x_test_file.close()
    y_test_file.close()


    # y_0 = len(y_train[y_train == 0])/len(y_train)
    # y_1 = 1 - y_0
    #
    # svm_clf = SVC(class_weight={0:y_1, 1:y_0})
    # svm_clf.fit(X_train_R, y_train)
    #
    # svm_predictions = svm_clf.predict(X_test_R) # Save prediction
    #
    #
    # scores = cross_val_score(svm_clf, X_test_R, y_test, cv=5)
    # print(scores)
    # print("Accuracy: %0.2f (+/- %0.2f)\n" % (scores.mean(), scores.std() * 2))
    #
    # print(mt.classification_report(y_test, svm_predictions))
    # print(mt.confusion_matrix(y_test, svm_predictions))
    # print(df.shape)
    # print('done')


if __name__ == "__main__":
    pre_process_loan_data('../')
