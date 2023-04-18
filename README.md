# my-neural-fields

### Build & Run docker

- `docker build -t neural-fields .` - to build docker
- To run a container, use:
```
docker run -d -it --init \
--gpus=all \
--ipc=host \
--volume="$PWD:/app" \
--volume="/home/jupyter/data:/data" \
--volume="/usr/local/cuda:/usr/local/cuda" \
--publish="7771:7771" \
--publish="7772:7772" \
neural-fields bash
```
- To execute container bash interface, run: `docker exec -it <container_id> bash` 

#### Run jupyter server

- `jupyter lab --no-browser --ip 0.0.0.0 --port 7771 --allow-root --notebook-dir=.`

- `pip install tensorboard`
- `tensorboard --logdir=logs --port 7773`
