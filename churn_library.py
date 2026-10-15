'''
EDA then train and test the models.

Author: Mutab
Date: 10/16/2026
'''

import os

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

os.environ["QT_QPA_PLATFORM"] = "offscreen"

EDA_DIR = "./images/eda"
RESULTS_DIR = "./images/results"
MODELS_DIR = "./models"
DATA_PATH = "./data/bank_data.csv"


def create_output_directories():
    """
    Create output directories used by the project.

    input:
            None
    output:
            None
    """
    os.makedirs(EDA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def import_data(pth):
    """
    Return a dataframe for the csv found at pth.

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    return df


def perform_eda(df):
    """
    Perform EDA on df and save figures.

    input:
            df: pandas dataframe
    output:
            None
    """
    create_output_directories()

    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(f"{EDA_DIR}/key_dist_plot.png", dpi=300, bbox_inches="tight")

    plt.figure(figsize=(20, 10))
    corr = df.select_dtypes(include='number').corr()
    sns.heatmap(corr, annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(
        f"{EDA_DIR}/correlation_heatmap.png",
        dpi=300,
        bbox_inches="tight")


def encoder_helper(df, category_lst, response):
    """
    Encode categorical features.

    input:
            df: pandas dataframe
            category_lst: list of categorical columns
            response: response column name
    output:
            df: updated dataframe
    """
    for category in category_lst:
        df[f'{category}_{response}'] = df[category].map(
            df.groupby(category)[response].mean()
        )

    return df


def perform_feature_engineering(df, response):
    """
    Split dataset into train and test sets.

    input:
              df: pandas dataframe
              response: response column name
    output:
              x_train, x_test, y_train, y_test
    """
    y = df[response]
    X = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(
    y_train,
    y_test,
    y_train_preds_lr,
    y_train_preds_rf,
    y_test_preds_lr,
    y_test_preds_rf,
):
    """
    Save classification reports as images.

    input:
            predictions and labels
    output:
            None
    """
    create_output_directories()

    output_text = (
        "random forest results\n"
        "test results\n"
        f"{classification_report(y_test, y_test_preds_rf)}\n"
        "train results\n"
        f"{classification_report(y_train, y_train_preds_rf)}"
    )

    # Create a blank figure canvas
    _, ax = plt.subplots(figsize=(7, 7))

    # Hide the chart axes, borders, and ticks so it's just a clean sheet
    ax.axis("off")

    # Render the text onto the canvas
    ax.text(
        0.01,
        0.99,  # Coordinates (X=left margin, Y=top margin)
        output_text,
        fontsize=10,
        family="monospace",  # Keeps your metrics tables perfectly aligned
        va="top",  # Vertically aligns text to the top
        ha="left",  # Horizontally aligns text to the left
    )

    # Save the canvas as an image
    plt.savefig(
        f"{RESULTS_DIR}/rf_classification_results.png",
        dpi=300,
        bbox_inches="tight")

    output_text = (
        "logistic regression results\n"
        "test results\n"
        f"{classification_report(y_test, y_test_preds_lr)}\n"
        "train results\n"
        f"{classification_report(y_train, y_train_preds_lr)}"
    )

    _, ax = plt.subplots(figsize=(7, 7))

    ax.axis("off")

    ax.text(
        0.01,
        0.99,  # Coordinates (X=left margin, Y=top margin)
        output_text,
        fontsize=10,
        family="monospace",  # Keeps your metrics tables perfectly aligned
        va="top",  # Vertically aligns text to the top
        ha="left",  # Horizontally aligns text to the left
    )

    plt.savefig(
        f"{RESULTS_DIR}/lr_classification_results.png",
        dpi=300,
        bbox_inches="tight")


def feature_importance_plot(model, x_data, output_pth):
    """
    Save feature importance plot.

    input:
            model, x_data, output path
    output:
            None
    """
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(x_data)

    shap_list = [
        shap_values[:, :, 0],
        shap_values[:, :, 1]
    ]

    shap.summary_plot(shap_list, x_data, plot_type="bar", show=False)
    ax = plt.gca()
    ax.set_xlabel('Mean | SHAP value | (feature importance)')
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        ["Non-Churn (0)", "Churn (1)"],
        loc="lower right"
    )
    plt.savefig(
        f"{RESULTS_DIR}/feature_importance_plot.png",
        dpi=300,
        bbox_inches="tight")


def train_models(x_train, x_test, y_train, y_test):
    """
    Train models and save outputs.

    input:
            train/test data
    output:
            None
    """
    create_output_directories()

    # grid search
    rfc = RandomForestClassifier(random_state=42)
    # Use a different solver if the default 'lbfgs' fails to converge
    # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
    # lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=3000))
    ])

    param_dist = {
        'n_estimators': [200, 300],
        'max_features': ['sqrt'],
        'max_depth': [5, 8, 10],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4],
        'criterion': ['gini']
    }

    cv_rfc = RandomizedSearchCV(
        estimator=rfc,
        param_distributions=param_dist,
        n_iter=12,
        cv=3,
        random_state=42,
        n_jobs=-1,
        error_score='raise'
    )

    cv_rfc.fit(x_train, y_train)
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    feature_importance_plot(cv_rfc, x_test, RESULTS_DIR)

    joblib.dump(cv_rfc, f"{MODELS_DIR}/random_forest_model.pkl")


if __name__ == "__main__":
    create_output_directories()

    df = import_data(DATA_PATH)

    perform_eda(df)

    category_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]

    df = encoder_helper(df, category_columns, "Churn")
    x_train, x_test, y_train, y_test = perform_feature_engineering(df, "Churn")
    train_models(x_train, x_test, y_train, y_test)
