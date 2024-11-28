import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

project_path = r"C:\Users\MINER5\Desktop\School\WGU\Machine Learning DevOps\Deploying-a-Scalable-ML-Pipeline-with-FastAPI" 

#Load cencus.csv data
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)
data = pd.read_csv(data_path)

# Split the data into train and test datasets
train, test = train_test_split(data, test_size=0.20, random_state=42)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# process the training data.
X_train, y_train, encoder, lb = process_data(
    X=train,
    categorical_features=cat_features,
    label="salary",
    training=True
    )

#process the test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model
print("Training the model...")
model = train_model(X_train, y_train)

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)

# load the model
model = load_model(
    model_path
) 

# Run model inferences on the test dataset
print("Running inference on the test dataset...")
preds = inference(model, X_test)

# Compute the performance on model slices
slice_output_path = os.path.join(project_path, "slice_output.txt")

for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):  # `slicevalue` is the correct variable name
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
           data=test,
           column_name=col,
           slice_value=slicevalue,
           categorical_features=cat_features,
           label="salary",
           encoder=encoder,
           lb=lb,
           model=model
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
