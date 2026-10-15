# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree (Udacity)

---

## Project Description

> We did an EDA on the data, then we trained 2 models (Random forest, and logstic regression) nad we compare the results of both of them and save the models so we can choose the one with the best result.

---

## Files and Data Description

> TODO:
> In the /images we stored the figures and plots for the eda and model results.
> In the /models we save the trained models, you will find the lr and rf models in there.
> in the /logs we store the logs for running the testing script "churn_script_logging_and_tests.py"

### Main Files

- `churn_library.py`  
  > EDA then train, test and save the models.

- `churn_script_logging_and_tests.py`  
  > This module contains tests for the functions in 'churn_library.py'

- `churn_notebook.ipynb`  
  > contains the data science and ML part but in a notebook that is not ready to be deployed for production.

---

### Data

- `data/bank_data.csv`  
  > We used the bank_data to identify credit card customers who are most likely to churn, by using the most importance feautures after th EDA.

---

### Output Directories

After running the project, outputs will be saved to:

- EDA images → `images/eda/`
- Model results → `images/results/`
- Models → `models/`
- Logs → `logs/churn_library.log`

---

## Running the Files

### 1. Run the Pipeline

```bash
python churn_library.py
```

> TODO:
> It will pre-proccess the data, do some EDA, train and test the models, compare the results and save the models and there results.

---

### 2. Run Tests and Logging

```bash
python churn_script_logging_and_tests.py
```

> TODO:
> It will run test cases to make sure 'churn_library.py' works as expected and store the log of the tests in /logs.

---

## Expected Outputs

1- Images for the EDA and model results.
2- The trained models.
3- the test logs.

Minimum expected outputs:

- Models:
  - `models/rfc_model.pkl`
  - `models/logistic_model.pkl`

- Images:
  - EDA plots in `images/eda/`
  - Model evaluation plots in `images/results/`

- Logs:
  - `logs/churn_library.log`

---