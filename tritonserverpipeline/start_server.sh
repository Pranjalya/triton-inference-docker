cd /home/azureuser/src/tritonserverpipeline; sudo docker run -d  --restart unless-stopped --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size 24G -v $PWD/models:/models -v $PWD/cache:/cache tritonserverpipeline:latest