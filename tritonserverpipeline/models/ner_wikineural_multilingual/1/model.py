import os
import json
import torch
import triton_python_backend_utils as pb_utils
import numpy as np
os.environ['HF_HOME'] = '/cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
from transformers import pipeline


class TritonPythonModel:
    def initialize(self, args):
        device = 0 if torch.cuda.is_available() else -1
        self.classifier = pipeline("ner", model="Babelscape/wikineural-multilingual-ner", aggregation_strategy="simple", device=device)

    def execute(self, requests):
        responses = []
        for request in requests:
            query = [t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "text").as_numpy().tolist()]

            # Call the Model pipeline
            pipeline_output = self.classifier(query)
            # Convert `score` key from numpy.float32 to Python float to make it JSON serializable
            ner_results = [[{k:float(v) if k=='score' else v for k,v in entity.items()} for entity in item] for item in pipeline_output]
            outputs = list()
            for result in ner_results:
                outputs.append(pb_utils.Tensor("result", np.array([json.dumps(result).encode("UTF-8")])))

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses

    def finalize(self):
        self.classifier = None