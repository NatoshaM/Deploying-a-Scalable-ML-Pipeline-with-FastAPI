import json

import requests

# TODO: send a GET using the URL http://127.0.0.1:8000
url = "http://127.0.0.1:8000"

# Test GET request
try:
    r = requests.get(url)
    print("GET Status Code:", r.status_code)
    print("GET Response Message:", r.json())
except requests.exceptions.JSONDecodeError:
    print("Failed to decode JSON from GET response.")
except Exception as e:
    print(f"Error during GET request: {e}")

# Define the payload for POST request
data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Test POST request
try:
    r = requests.post(f"{url}/data/", json=data)
    print(f"POST Status Code: {r.status_code}")
    if r.status_code == 200:  # Check for successful status
        response_json = r.json()
        print(f"Prediction Result: {response_json.get('result', 'No result key in response')}")
    else:
        print(f"Error: Server returned status code {r.status_code}. Response content: {r.content}")
except requests.exceptions.JSONDecodeError:
    print(f"Failed to decode JSON. Response content: {r.content}")
except Exception as e:
    print(f"Error during POST request: {e}")
