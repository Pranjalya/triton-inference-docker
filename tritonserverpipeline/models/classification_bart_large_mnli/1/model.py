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
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

    def execute(self, requests):
        responses = []
        for request in requests:
            query = [t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "text").as_numpy().tolist()]
            
            labels = pb_utils.get_input_tensor_by_name(request, "labels")
            candidate_labels = labels.as_numpy()[0].decode("UTF-8").split("|")
            candidate_labels = list(filter(lambda x: len(x.strip()), candidate_labels))

            multi_label = pb_utils.get_input_tensor_by_name(request, "multi_label")
            is_multi_label = bool(multi_label.as_numpy()[0])
        
            # Call the Model pipeline
            pipeline_output = self.classifier(query, candidate_labels, multi_label=multi_label)
            results = []
            for po in pipeline_output:
                result = {}
                for i, label in enumerate(po["labels"]):
                    result[label] = po["scores"][i]
                results.append(result)

            inference_response = pb_utils.InferenceResponse(
			output_tensors=[
				pb_utils.Tensor(
					"result",
					np.array([json.dumps(results).encode("UTF-8")]),
					)
				]
			)
            responses.append(inference_response)

        return responses

    def finalize(self):
        self.classifier = None