import requests

# Define the URL and the payload
url = "http://localhost:8000/chat/"
payload = {
    "messages": [
        {"role": "system", "content": "You're an angry pirate captain who always responds in pirate speak."},
        {"role": "assistant", "content": "How are you doing today?"},
    ],
    "max_new_tokens": 512,
    "temperature": 0.5
}

# Send the POST request
response = requests.post(url, json=payload)

# Print the response
print(response.json())