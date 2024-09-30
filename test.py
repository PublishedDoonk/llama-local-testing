import requests
import timeit

def test():
    # Define the URL and the payload
    url = "http://localhost:8000/chat/"
    payload = {
        "messages": [
            {"role": "system", "content": "You're an angry pirate captain who always responds in pirate speak."},
            {"role": "user", "content": "Why do you sound so goofy?"},
        ],
        "max_new_tokens": 512,
        "temperature": 0.5
    }

    # Send the POST request
    response = requests.post(url, json=payload)

    # Print the response
    print(response.json()['messages'])

if __name__ == "__main__":
    test()
    print("testing 10 times for execution time")
    execution_time = timeit.timeit(test, number=10)
    print(f"Average execution time over 10 runs: {execution_time / 10:.6f} seconds")

