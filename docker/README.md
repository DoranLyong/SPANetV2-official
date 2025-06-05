# Docker Container Settings 
You can refer to the list of NVIDIA Docker images [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) and [NGC catalog](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags).
An example of a container list:
```bash
# -- pt1.13-cu11.8
nvcr.io/nvidia/pytorch:22.11-py3

# -- pt2.10-cu12.2
nvcr.io/nvidia/pytorch:23.10-py3
```



Construcsts and runs a docker container using `make_container.sh` and `run_container.sh`
```bash
# -- Step1
bash make_container.sh

# -- Step2
bash run_container.sh
```

You can change the volume path by editing:
```bash 
#  <local_volume>:<docker_volume>
-v $HOME/Workspace:/workspace
```

You can change the number of CPU cores by editing:
```bash
--cpuset-cpus=0-24
```

