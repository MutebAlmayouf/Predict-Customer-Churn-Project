'''
This module contains tests for the functions in 'churn_library.py'.

Author: Muteb.
Date created: 10/15/2026
'''

import logging
import os
from pathlib import Path

import churn_library as cls

LOGS_DIR = "./logs"
LOG_FILE = os.path.join(LOGS_DIR, "churn_library.log")
DATA_PATH = "./data/bank_data.csv"

os.makedirs(LOGS_DIR, exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s'
)


def test_import(import_data):
    """Test data import."""
    try:
        df = import_data(DATA_PATH)

        logging.info('SUCCESS: Data import done.')

    except FileNotFoundError as err:
        logging.error('ERROR: File for importing data is not found.')
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0

    except AssertionError as err:
        logging.error('ERROR: Data is empty.')
        raise err


def test_eda(perform_eda):
    """Test EDA."""
    try:
        df = cls.import_data(DATA_PATH)
        perform_eda(df)

        folder = Path("./images/eda")

        # Asserts the folder exists, is a directory, and is not empty
        assert folder.is_dir(), f"'{folder}' is not a valid directory."
        assert any(folder.iterdir()), f"The folder '{folder}' is empty."

        logging.info('SUCCESS: EDA performed.')

    except Exception:
        logging.error('ERROR: EDA output files doesn\'t exist.')


def test_encoder_helper(encoder_helper):
    """Test encoding."""
    try:
        df = cls.import_data(DATA_PATH)

        category_lst = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ]
        response = 'Churn'
        df = encoder_helper(df, category_lst, response)

        assert "Churn" in df.columns, "The 'churn' column is missing from the DataFrame."

        logging.info('SUCCESS: Encoding the column successfully.')

    except Exception:
        logging.error('ERROR: Encoded column doesn\'t exist.')


def test_perform_feature_engineering(
        perform_feature_engineering,
        encoder_helper):
    """Test feature engineering."""
    try:
        df = cls.import_data(DATA_PATH)

        category_lst = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ]
        response = 'Churn'
        df = encoder_helper(df, category_lst, response)

        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, response)

        train_size = 0.7
        total_size = df.shape[0]
        assert len(x_train) == int(total_size * train_size)
        assert len(x_test) == total_size - int(total_size * train_size)

        assert len(y_train) == int(total_size * train_size)
        assert len(y_test) == total_size - int(total_size * train_size)

        logging.info('SUCCESS: Feature engineering done successfully.')

    except Exception:
        logging.error('ERROR: Feature Engineering size mismatch.')



def test_train_models(
        train_models,
        encoder_helper,
        perform_feature_engineering):
    """Test model training."""
    try:
        df = cls.import_data(DATA_PATH)

        category_lst = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category",
        ]
        response = 'Churn'
        df = encoder_helper(df, category_lst, response)

        x_train, x_test, y_train, y_test = perform_feature_engineering(
            df, response)

        train_models(x_train, x_test, y_train, y_test)

        images_folder = Path("./images/results")
        models_folder = Path("./models")

        assert images_folder.is_dir(
        ), f"'{images_folder}' is not a valid directory."
        assert any(images_folder.iterdir()
                   ), f"The folder '{images_folder}' is empty."

        assert models_folder.is_dir(
        ), f"'{models_folder}' is not a valid directory."
        assert any(models_folder.iterdir()
                   ), f"The folder '{models_folder}' is empty."

        # TODO: logging success
        logging.info('SUCCESS: Test and train models successfully.')

    except Exception:

        # TODO: logging failure
        logging.error('ERROR: Test and train model unseccessful.')



if __name__ == "__main__":
    assert Path(LOGS_DIR).is_dir(), f'{LOGS_DIR} doesn\'t exist'

    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(
        cls.perform_feature_engineering,
        cls.encoder_helper)
    test_train_models(
        cls.train_models,
        cls.encoder_helper,
        cls.perform_feature_engineering)

    print("Tests completed. Check logs for details.")
