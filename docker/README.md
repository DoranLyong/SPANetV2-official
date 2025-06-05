# Docker Container Settings 
You can refer to the list of NVIDIA Docker images [here](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html).


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

