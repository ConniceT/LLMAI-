
import requests
response = requests.post("https://example.com/items", json={"name": "John Doe", "age": 30})
print(response.json())
