FROM nvcr.io/nvidia/tritonserver:23.06-py3

# Install required packages
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y \
    curl

# Install transformers and sentencepiece
RUN pip install transformers sentencepiece

# Set the model repository path
ENV MODEL_REPOSITORY=/models

# Expose necessary ports
EXPOSE 8000 8001 8002

# Set the shared memory size
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set the shared memory size
ENV CUDA_VISIBLE_DEVICES=all
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV CUDA_CACHE_PATH=/dev/null

# Set the shared memory size
ENV TRITONSERVER_SHM_BYTE_LIMIT=268435456

# Set the entrypoint command
CMD ["bash", "-c", "tritonserver --model-repository=$MODEL_REPOSITORY"]