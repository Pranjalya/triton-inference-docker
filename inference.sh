# CPU
docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size 256m -v $PWD/triton_models/transformer_onnx:/models -it  triton-inference-server

# GPU
docker run  --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size 256m -v $PWD/triton_models/transformer_onnx:/models -it  triton-inference-server