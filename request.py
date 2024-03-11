import requests

# Replace this URL with your API endpoint
api_url = "http://127.0.0.1:5000/"  # Replace with your actual endpoint

# Example input data (replace with your own data)
input_data = {
    "state": "East Rajasthan",
    "month": 1,
    "year": 2023
}

# Send a POST request to the API endpoint
response = requests.post(api_url, json=input_data)

# Print the response
print("Response:", response.json())
