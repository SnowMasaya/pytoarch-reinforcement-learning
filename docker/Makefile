help:
	@cat Makefile

DATA?="${HOME}/Data"
GPU?=0
DOCKER_FILE=Dockerfile
DOCKER_FILE_GPU=Dockerfile
DOCKER=docker
DOCKER_GPU=GPU=$(GPU) nvidia-docker
TEST=tests/
SRC=$(shell dirname `pwd`)
CUDA=/usr/local/cuda
LD_LIBRARY=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64

# CPU environment

build:
	$(DOCKER) build -t deep-learning-with-pytoarch --build-arg python_version=3.5 -f $(DOCKER_FILE) .

bash: build
	$(DOCKER) run -p 8889:8889 -p 6007:6007 -it -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix -v $(SRC):/workspace -v $(DATA):/data deep-learning-with-pytoarch bash

ipython: build
	$(DOCKER) run -p 8888:8888 -p 6006:6006 -it -v $(SRC):/workspace -v $(DATA):/data deep-learning-with-pytoarch ipython

notebook: build
	$(DOCKER) run -p 8888:8888 -p 6006:6006 -it -v $(SRC):/workspace -v $(DATA):/data deep-learning-with-pytoarch

test: build
	$(DOCKER) run -it -v $(SRC):/workspace -v $(DATA):/data deep-learning-with-pytoarch pytest $(TEST)

# GPU environment

build-gpu:
	$(DOCKER_GPU) build -t deep-learning-with-pytoarch-gpu --build-arg python_version=3.5 -f $(DOCKER_FILE_GPU) .

bash-gpu: build-gpu
	$(DOCKER_GPU) run -p 8889:8889 -p 6007:6007 -it -v $(SRC):/workspace -v $(DATA):/data \
	                                                --env CUDA_HOME=${CUDA} \
	                                                --ipc=host \
	                                                --net=host \
	                                                -e DISPLAY=${DISPLAY} \
	                                                -v /tmp/.X11-unix:/tmp/.X11-unix \
	                                                --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                deep-learning-with-pytoarch-gpu bash
ipython-gpu: build-gpu
	$(DOCKER_GPU) run -p 8888:8888 -p 6006:6006  -it -v $(SRC):/workspace -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  deep-learning-with-pytoarch-gpu ipython

notebook-gpu: build-gpu
	$(DOCKER_GPU) run -p 8888:8888 -p 6006:6006 -it -v $(SRC):/workspace -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  deep-learning-with-pytoarch-gpu

test-gpu: build-gpu
	$(DOCKER_GPU) run -it -v $(SRC):/workspace -v $(DATA):/data --env CUDA_HOME=${CUDA} \
	                                                  --env LD_LIBRARY_PATH=${LD_LIBRARY} \
	                                                  deep-learning-with-pytoarch-gpu pytest $(TEST)

