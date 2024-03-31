import time
import requests
import random
import json
# from tqdm import tqdm

headers = {
    'Content-Type': 'application/json',
}

sentence = """one day I will see the world"""
candidate_labels = ['travel', 'cooking', 'dancing']

start = time.time()

json_data = {
    'inputs': [
        {
            'name': 'text',
            'shape': [2],
            'datatype': 'BYTES',
            'data': [sentence, sentence],
        },
        {
            'name': 'labels',
            'shape': [1],
            'datatype': 'BYTES',
            'data': ["|".join(candidate_labels)],
        },
        {
            'name': 'multi_label',
            'shape': [1],
            'datatype': 'BOOL',
            'data': [False],
        },
    ],
}

response = requests.post(
    'http://localhost:8000/v2/models/classification_bart_large_mnli/versions/1/infer',
    headers=headers,
    json=json_data,
)
if "error" in response.json():
    print(response.json())

result = [json.loads(output['data'][0]) for output in response.json()['outputs']]

print(result)

print("Total time : ", time.time() - start)