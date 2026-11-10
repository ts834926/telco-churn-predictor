# Telco Customer Churn Predictor

A beginner-friendly project that takes one customer CSV, explores it, compares several churn models, and serves the selected model through a small API.

## Start here

Use **`notebooks/telco_churn_project.ipynb`** as the complete modeling workflow. It includes data checks, cleaning, a single train/test split, cross-validation, hyperparameter tuning, model comparison, ensemble learning, and a final test-set evaluation. The cross-validation folds and tuning happen only on the training portion; the held-out test portion is used once at the end.

The notebook saves the fitted preprocessing and prediction pipeline as `models/churn_model.joblib`. The API and Docker setup use this saved model.

## Files used for this project

```text
data/telco_churn.csv                    Dataset
notebooks/telco_churn_project.ipynb     Complete modeling workflow
models/churn_model.joblib               Model created by the notebook
api/app.py                              Prediction API
api/test_app.py                         API checks
Dockerfile                              Container setup
docker-compose.yml                      One-command API setup
requirements.txt                        Python packages
README.md                               This guide
```

The project folder keeps the data, notebook, model, and API code in separate folders. You only need to run the one notebook shown above.

## Set up and run the notebook (Windows)

Open PowerShell in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name telco-churn --display-name "Python (Telco Churn)"
.\.venv\Scripts\jupyter-lab.exe
```

Open `notebooks/telco_churn_project.ipynb` and run its cells from top to bottom. Select the **Python (Telco Churn)** kernel so it uses the packages in `.venv`, including the scikit-learn version used to save the model. The notebook reads `data/telco_churn.csv`. If the dataset is missing, get the IBM Telco Customer Churn sample from [IBM's public repository](https://github.com/IBM/telco-customer-churn-on-icp4d) or [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn), and save it in the `data` folder with that filename.

## What the notebook covers

1. Checks the 7,043-row sample and cleans blank `TotalCharges` values for zero-tenure customers.
2. Makes one stratified 80/20 train/test split.
3. Compares baseline models using stratified cross-validation.
4. Tunes Logistic Regression, Decision Tree, and Random Forest settings with cross-validation.
5. Compares tuned Random Forest with Gradient Boosting and a soft-voting ensemble.
6. Chooses the model using training-set CV F1, evaluates it once on the held-out test set, and saves it as `models/churn_model.joblib`.

In the included run, the selected tuned Random Forest reached **0.631 F1** and **0.844 ROC-AUC** on the held-out test set. These are results for one split of a static educational dataset, not a promise about future customers or retention campaigns.

## Run the API

The API expects all 19 customer features used by the model. It returns a Yes/No prediction, churn and retention probabilities, and a risk band (Low below 0.30, Medium from 0.30 to below 0.60, High at 0.60 or above).

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API page. The main routes are `GET /`, `GET /health`, and `POST /predict`.

## Run with Docker

After Docker Desktop is installed and running, build and start the API from this folder:

```powershell
docker compose up --build
```

The API will be available at `http://localhost:8000/docs`. Stop it with `Ctrl+C`, then run `docker compose down`.

## Check the API

```powershell
.\.venv\Scripts\python.exe -m pytest -q api/test_app.py
```

The dataset is a fictional, static sample. A real future-churn system should use information recorded before the prediction date and should validate decisions with a retention experiment and business costs.
