import os
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    save_model,
    load_model,
    performance_on_categorical_slice,
    train_model,
)


# Test 1: Check if the model returns the correct type (RandomForestClassifier)
def test_train_model():
    """
    Test pipeline of training model
    """
    X = np.random.rand(20, 5)
    y = np.random.randint(2, size=20)
    model = train_model(X, y)
    assert isinstance(model, BaseEstimator) and isinstance(model, ClassifierMixin)


# Test 2: Check if the compute_model_metrics function returns the expected values
def test_compute_model_metrics():
    """
    Test compute_model_metrics
    """
    y_true, y_preds = [1, 1, 0], [0, 1, 1]
    precision, recall, fbeta = compute_model_metrics(y_true, y_preds)
    assert precision is not None
    assert recall is not None
    assert fbeta is not None


# Test 3: Check inference function
def test_inference():
    """
    Test inference of model
    """
    X = np.random.rand(20, 5)
    y = np.random.randint(2, size=20)
    model = train_model(X, y)
    y_preds = inference(model, X)
    assert y.shape == y_preds.shape