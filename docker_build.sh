#!/usr/bin/env bash
if [[ "$(< /proc/version)" == *@(Microsoft|WSL)* ]]; then
  if [ "$(sudo service docker status)" = " * Docker is not running" ]; then
    sudo service docker start
  fi
else
  if [ "$(sudo systemctl is-active docker)" = "inactive" ]; then
    sudo systemctl restart docker
  fi
fi
IMAGE_LABEL=axd465/pilab_tensorflow:gpu-latest
cd ./docker_build_context
# docker build --rm -t axd465/pilab_tensorflow:gpu-latest
# docker build --progress=plain --rm -t $IMAGE_LABEL .
# docker build --compress --rm -t $IMAGE_LABEL .
docker build --rm -t $IMAGE_LABEL .
cd ..
port_number=9999 #8888 # Starting Port
while nc -z localhost $port_number ; do
  ((port_number++))
done
docker run -it --rm $@ -u root -p $port_number:$port_number -e \
JUPYTER_PORT=$port_number -v "${PWD}":/tf $IMAGE_LABEL
