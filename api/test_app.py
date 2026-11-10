"""Small API checks that can run locally or inside Docker."""

import pytest
from fastapi.testclient import TestClient

from api.app import MODEL_VERSION, app, get_risk_category, load_model


CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["project"] == "Telco Customer Churn Predictor"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_swagger_docs(client):
    assert client.get("/docs").status_code == 200


def test_valid_prediction_response(client):
    response = client.post("/predict", json=CUSTOMER)
    assert response.status_code == 200
    assert set(response.json()) == {
        "churn_prediction", "churn_probability", "retention_probability",
        "risk_category", "model_version",
    }
    assert response.json()["model_version"] == MODEL_VERSION


def test_missing_field_returns_422(client):
    customer = CUSTOMER.copy()
    customer.pop("tenure")
    assert client.post("/predict", json=customer).status_code == 422


def test_invalid_category_returns_422(client):
    customer = CUSTOMER | {"Contract": "Weekly"}
    assert client.post("/predict", json=customer).status_code == 422


@pytest.mark.parametrize("field,value", [("tenure", -1), ("MonthlyCharges", 250), ("SeniorCitizen", 2)])
def test_invalid_numbers_return_422(client, field, value):
    customer = CUSTOMER | {field: value}
    assert client.post("/predict", json=customer).status_code == 422


def test_model_file_loads():
    model = load_model()
    assert list(model.classes_) == [0, 1]


def test_probabilities_are_valid_and_add_to_one(client):
    result = client.post("/predict", json=CUSTOMER).json()
    assert 0 <= result["churn_probability"] <= 1
    assert 0 <= result["retention_probability"] <= 1
    assert result["churn_probability"] + result["retention_probability"] == pytest.approx(1, abs=0.0001)


@pytest.mark.parametrize(
    "probability,expected",
    [(0.0, "Low"), (0.2999, "Low"), (0.30, "Medium"), (0.5999, "Medium"), (0.60, "High"), (1.0, "High")],
)
def test_risk_category_boundaries(probability, expected):
    assert get_risk_category(probability) == expected


def test_prediction_risk_category_matches_probability(client):
    result = client.post("/predict", json=CUSTOMER).json()
    assert result["risk_category"] == get_risk_category(result["churn_probability"])
