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
        self.classifier = pipeline("text-classification", model="lxyuan/distilbert-base-multilingual-cased-sentiments-student", device=device)
        

    def execute(self, requests):
        responses = []
        for request in requests:
            query = [t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "text").as_numpy().tolist()]

            # Call the Model pipeline
            pipeline_output = self.classifier(query)

            outputs = list()
            for result in pipeline_output:
                outputs.append(pb_utils.Tensor("result", np.array([json.dumps(result).encode("UTF-8")])))

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses

    def finalize(self):
        self.classifier = None