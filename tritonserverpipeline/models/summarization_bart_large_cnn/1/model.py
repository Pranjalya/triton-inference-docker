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
        self.generator = pipeline("summarization", model="facebook/bart-large-cnn", device=device)

    def execute(self, requests):
        responses = []
        for request in requests:
            query = [t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "text").as_numpy().tolist()]

            # Call the Model pipeline
            pipeline_output = self.generator(query, do_sample=True)
            generated_txts = [po["summary_text"] for po in pipeline_output]
            outputs = list()
            for generated_txt in generated_txts:
                outputs.append(pb_utils.Tensor("summary_text", np.array([generated_txt.encode("UTF-8")])))

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses

    def finalize(self):
        self.generator = None