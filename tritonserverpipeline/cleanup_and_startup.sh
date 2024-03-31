mkdir -p cache
sudo docker ps -a | grep Exit | cut -d ' ' -f 1 | xargs sudo docker rm
sudo docker run -d --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size 24G -v $PWD/models:/models -v $PWD/cache:/cache tritonserverpipeline:latest
# docker run  --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 --shm-size 16384m -v $PWD/models:/models -v $PWD/cache:/cache tritonserverpipeline:latest