import os
import json
import torch
import triton_python_backend_utils as pb_utils
import numpy as np
os.environ['HF_HOME'] = '/cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
from gliner import GLiNER


class TritonPythonModel:
    def initialize(self, args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = GLiNER.from_pretrained("urchade/gliner_multi")
        self.model.to(device)

    def execute(self, requests):
        responses = []
        for request in requests:
            query = [t.decode("UTF-8") for t in pb_utils.get_input_tensor_by_name(request, "text").as_numpy().tolist()]

            labels = pb_utils.get_input_tensor_by_name(request, "labels")
            candidate_labels = labels.as_numpy()[0].decode("UTF-8").split("|")
            candidate_labels = list(filter(lambda x: len(x.strip()), candidate_labels))

            # Call the Model pipeline
            ner_results = [self.model.predict_entities(q, candidate_labels, threshold=0.5) for q in query]
            outputs = list()
            for result in ner_results:
                outputs.append(pb_utils.Tensor("result", np.array([json.dumps(result).encode("UTF-8")])))

            inference_response = pb_utils.InferenceResponse(output_tensors=outputs)
            responses.append(inference_response)

        return responses

    def finalize(self):
        self.model = None