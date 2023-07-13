# triton-inference-docker
Docker image to run a Sentence Transformer / Transformer model in NVIDIA Triton Inference.

Steps : 
1. 
```
git clone https://github.com/ELS-RD/transformer-deploy
cd transformer-deploy
```

2.
```
pip3 install ".[GPU]" -f https://download.pytorch.org/whl/cu113/torch_stable.html --extra-index-url https://pypi.ngc.nvidia.com
# if you want to perform GPU quantization (recommended):
pip3 install git+https://github.com/NVIDIA/TensorRT#egg=pytorch-quantization\&subdirectory=tools/pytorch-quantization/
# if you want to accelerate dense embeddings extraction:
pip install sentence-transformers
```