import pytest
import pandas as pd
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
    Test if the trained model is of type RandomForestClassifier.
    """
    # Example data for testing (this should be a subset or mock data in reality)
    X_train = pd.DataFrame({'age': [25, 30, 35, 40], 'hours-per-week': [40, 50, 60, 70]})
    y_train = [0, 1, 0, 1]

    model = train_model(X_train, y_train)
    
    # Check if the model is a RandomForestClassifier
    assert isinstance(model, RandomForestClassifier), f"Expected RandomForestClassifier, got {type(model)}"



# TODO: implement the second test. Change the function name and input as needed

# Test 2: Check if the compute_model_metrics function returns the expected values (precision, recall, fbeta)
def test_compute_model_metrics():
    """
    Test the compute_model_metrics function to ensure it returns the correct values.
    """
    # Example data for testing (mock predictions and true values)
    y_true = [1, 0, 1, 1, 0]
    y_pred = [1, 0, 1, 0, 0]

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    # Check if the metrics are floats
    assert isinstance(precision, float), f"Expected float for precision, got {type(precision)}"
    assert isinstance(recall, float), f"Expected float for recall, got {type(recall)}"
    assert isinstance(fbeta, float), f"Expected float for fbeta, got {type(fbeta)}"

# Test 3: Check if the performance_on_categorical_slice function works correctly (returns precision, recall, fbeta)
def test_performance_on_categorical_slice():
    """
    Test the performance_on_categorical_slice function to ensure it returns precision, recall, fbeta.
    """
    # Mock data for testing
    data = pd.DataFrame({
        'workclass': ['Private', 'Self-emp', 'Private', 'Self-emp'],
        'salary': ['<=50K', '>50K', '>50K', '<=50K'],
        'age': [25, 30, 35, 40],
        'hours-per-week': [40, 50, 60, 70]
    })
    
    # Example categorical features and label
    cat_features = ['workclass']
    label = 'salary'

    # Process the data (using mock encoder and lb)
    encoder = OneHotEncoder()
    lb = LabelBinarizer()
    encoder.fit(data[cat_features])
    lb.fit(data[label])

    X_slice, y_slice, _, _ = process_data(
        X=data,
        categorical_features=cat_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb
    )

    # Assume we are checking performance on 'Private' slice
    precision, recall, fbeta = performance_on_categorical_slice(
        data=data,
        column_name='workclass',
        slice_value='Private',
        categorical_features=cat_features,
        label=label,
        encoder=encoder,
        lb=lb,
        model=RandomForestClassifier()
    )

    # Check if the metrics are floats
    assert isinstance(precision, float), f"Expected float for precision, got {type(precision)}"
    assert isinstance(recall, float), f"Expected float for recall, got {type(recall)}"
    assert isinstance(fbeta, float), f"Expected float for fbeta, got {type(fbeta)}"
