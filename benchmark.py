import time
import requests
import random
from tqdm import tqdm

headers = {
    'Content-Type': 'application/json',
}

sentences = [
    "The sun is shining today.",
    "I love to go for long walks in the park.",
    "She sings beautifully.",
    "The cat chased the mouse.",
    "He enjoys playing the piano.",
    "The book is on the table.",
    "We went to the beach last weekend.",
    "They built a sandcastle on the shore.",
    "My favorite color is blue.",
    "I have a lot of work to do.",
    "The movie was really funny.",
    "She studied hard for the exam.",
    "The train arrived on time.",
    "He cooked a delicious meal.",
    "The flowers are in bloom.",
    "We visited a museum yesterday.",
    "I like to watch TV shows in my free time.",
    "They went on a hiking trip in the mountains.",
    "She wrote a poem for her friend.",
    "The soccer match ended in a tie.",
    "I need to buy groceries for the week."
]

start = time.time()
for i in tqdm(range(100)):
    j = random.randint(1, 15)
    json_data = {
        'inputs': [
            {
                'name': 'TEXT',
                'shape': [j],
                'datatype': 'BYTES',
                'data': random.sample(sentences, j),
            },
        ],
    }

    response = requests.post(
        'http://localhost:8000/v2/models/transformer_onnx_inference/versions/1/infer',
        headers=headers,
        json=json_data,
    )

print("Total time : ", time.time() - start)