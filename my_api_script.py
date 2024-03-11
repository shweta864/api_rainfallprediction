import requests
import json

# URL of your Flask API endpoint
api_url = "http://127.0.0.1:5000/predict_rainfall"

# Input data for the API request (replace with your actual data)
input_data = {
    'year': 2023,
    'month': 9
}

# Convert the input data to JSON format
json_data = json.dumps(input_data)

# Set the headers for the request
headers = {'Content-Type': 'application/json'}

# Make a POST request to the API endpoint
response = requests.post(api_url, data=json_data, headers=headers)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Parse the JSON response
    result = response.json()
    print("API Response:", result)
    # Access the predicted rainfall value
    predicted_rainfall = result['predicted_rainfall']
    print(f"Predicted Rainfall: {predicted_rainfall} mm")
else:
    print(f"Error: {response.status_code} - {response.text}")
