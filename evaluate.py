
import json
import logging
import os
import pickle
import tarfile

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


if __name__ == "__main__":
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path) as tar:
        tar.extractall(path="..")

    logger.debug("Loading linear regression model.")
    model = pickle.load(open("linear_regression_model", "rb"))

    print("Loading test input data")
    test_path = "/opt/ml/processing/test/validation_df.csv"
    df = pd.read_csv(test_path, header=None)

    logger.debug("Reading test data.")
    y_test = df.iloc[:, 0].to_numpy()
    df.drop(df.columns[0], axis=1, inplace=True)
    X_test = df.values

    logger.info("Performing predictions against test data.")
    predictions = model.predict(X_test)

    logger.info("Evaluating model performance.")
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Update report dictionary with chosen metrics
    report_dict = {
        "regression_metrics": {
            "mean_squared_error": {
                "value": mse,
                "standard_deviation": "NaN"
            },
            "r2_score": {
                "value": r2,
                "standard_deviation": "NaN"
            }
        }
    }

    print("Regression evaluation report:\n{}".format(report_dict))

    evaluation_output_path = os.path.join("/opt/ml/processing/evaluation", "evaluation.json")
    print("Saving evaluation report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report_dict))
