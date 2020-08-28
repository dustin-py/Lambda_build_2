import pandas as pd 
import numpy as np
from sklearn.metrics import accuracy_score, plot_confusion_matrix


def get_lowest_speed(X):
    """
    Get the lowest speed in the given range:
    1. Change df['dvcat'] values == '1-9km/h' so that they equal int(5)
    2. Instantiate an empty list to store lowest speeds
    3. Get the first 2 or 1 orccurances of the value
    4. Append those values to the lowest speed list
    5. Return a list of the lowest speeds
    """
    lowest_speeds = []
    for val in list(X.replace('1-9km/h', 5).dvcat.values):
        if len(str(val)) == 1:
            lowest_speeds.append(val)
        else:
            lowest_speeds.append(int(val[:2]))
    return lowest_speeds    

def get_highest_speed(X):
    """
    Get highest speed in the values range:
    1. Change df['dvcat'] values == '1-9km/h' so that they equal int(5)
    2. Instantiate empty list to store highest speed values
    3. Check is the length of the value as a string is > 3, if true then append the value
    4. Otherwise append a 0 to avoid duplicating the values without a range
    5. Return highest speed list
    """

    highest_speeds = []
    for val in list(X.replace('1-9km/h', 5).dvcat.values):
        if len(str(val)) > 3:
            highest_speeds.append(val[3:])
        else:
            highest_speeds.append(0)
    return highest_speeds    

def get_speed_difference(lowest_speeds, highest_speeds):
    """
    Get the difference in speed
    1. Instantiate an empty speed_diff list
    2. Subtract lowest_speeds from highest_speeds to get the difference
    3. Return the difference as a list 
    """
    iter_range = len(lowest_speeds)
    avg_speed = []
    for i in range(iter_range):
        result = int(lowest_speeds[i]) + (int(highest_speeds[i]) - int(lowest_speeds[i]))/2
        avg_speed.append(result)
    return avg_speed    


def load_in_data():
    original_X = pd.read_csv(
        "Data/nassCDS.csv", 
        index_col='Unnamed: 0', )
    return original_X


def wrangler(X):

    X.copy()

    target = 'injSeverity'

    # Define features to withhold from the data:
    cols_to_drop = ['weight', 'deploy', 'caseid']
    X = X.drop(columns=cols_to_drop)

    X[target] = X[target].map(
        {
            0.0: 'low',
            1.0: 'low',
            2.0: 'mid',
            3.0: 'mid',
            4.0: 'high',
            5.0: 'high',
            6.0: 'high',
        }
    )

    X = X.dropna(
        subset=['yearVeh', target]
    )

    for _ in range(3):
        X = X.append(X[X[target] == 'high'])   

    return X


def engineer_features(X):

    X.copy()

    X['yearVeh'] = X['yearVeh'].astype('int64')
    X['vehicle_age'] = X['yearacc'] - X['yearVeh']
    X['vehicle_age'] = X['vehicle_age'].astype('int64')
    X['year_occ_was_born'] = X['yearacc'] - X['ageOFocc']
    X['year_occ_got_license'] = X['year_occ_was_born'] + 16
    X['year_of_drive_exp'] = X['yearacc'] - X['year_occ_got_license']
    low_speed = get_lowest_speed(X),
    high_speed = get_highest_speed(X),
    X['low'] = low_speed[0]
    X['high'] = high_speed[0]
    X['avg_speed'] = get_speed_difference(
        low_speed[0],
        high_speed[0],
    )
#     X['avg_speed'] = X['avg_speed'].astype('int64')
#     X['speed_type'] = pd.cut(
#         X['avg_speed'],
#         [0,15,25,40,],
#         labels=[
#             'slow', 'slow-mid', 'fast-mid' 'fast',
#         ]
#     )
#     X['age_type'] = pd.cut(
#         X['ageOFocc'],
#         [0, 25, 55, 100],
#         labels=[
#             'young', 'mid-life', 'old',
#         ]
#     )
    X = pd.concat((X.drop(columns='high'), X.drop(columns='low')))
    X['speed'] = [ list(i) for i in [low_speed[0] + high_speed[0]]][0]
    X = X.drop(columns=['low', 'high'])

    return X


def split_data(X, return_test=False):
    from sklearn.model_selection import train_test_split
    X.copy()


    train, test = train_test_split(
        X,
        test_size=0.3,
        random_state=42, )

    train, val = train_test_split(
        train,
        test_size=0.3,
        random_state=42, )

    if return_test is True:
        return train, val, test
    else:
        return train, val


def make_model(X_train, y_train, X_test, y_test, model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    acc_score = accuracy_score(y_test, y_pred)
#     print(f"Classification Report: \n{classification_report(y_test, y_pred)}\n");
    print(f"Confusion Matrix: \n{plot_confusion_matrix(model, X=X_test, y_true=y_test)}");

    return model, acc_score, y_pred, y_prob