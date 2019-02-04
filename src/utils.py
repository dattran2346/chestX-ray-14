import pandas as pd
from constant import PATH, TRAIN_CSV, VAL_CSV, TEST_CSV


def get_chestxray_from_csv():
    result = []
    for f in [PATH/TRAIN_CSV, PATH/VAL_CSV, PATH/TEST_CSV]:
        df = pd.read_csv(f, sep=' ', header=None)
        images = df.iloc[:, 0].values
        labels = df.iloc[:, 1:].values
        result.append((images, labels))
    return result
