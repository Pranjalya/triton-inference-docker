import time
import requests
import random
import json
# from tqdm import tqdm

headers = {
    'Content-Type': 'application/json',
}

sentences = ["I am happy.", "I am sad."]

start = time.time()

json_data = {
    'inputs': [
        {
            'name': 'text',
            'shape': [2],
            'datatype': 'BYTES',
            'data': sentences,
        },
    ],
}

response = requests.post(
    'http://localhost:8000/v2/models/sentiment_distilbert_multilingual/versions/1/infer',
    headers=headers,
    json=json_data,
)
if "error" in response.json():
    print(response.json())

result = [json.loads(output['data'][0]) for output in response.json()['outputs']]

print(result)

print("Total time : ", time.time() - start)