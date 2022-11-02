# Installation and features

## User Installation
```shell
pip install plasticorigins
```

## Worker Installation

### Install Poetry
```shell
pip install poetry
```

### Create your virtual environment
Here we use python version 3.8
```shell
poetry use 3.8
```

### Install dependencies
```shell
poetry install
```

### Code Linting and Formatting:

Pre-commits have been added to format and check the linting of the code before any commit. This process will run:

- **PyUpgrade**: to make sure that the code syntax is up to date with the latest python versions
- **Black**: which is a code formatter 
- **Flake8**: to check that the code is properly formatted.

All this process is automatic to ensure the commited code quality. So as a good measure, prior to committing any code it is highly recommended to run:
```shell
poetry run black path/to/the/changed/code/directory(ies)
```
This will format the code that has been written and:
```shell
poetry run flake8 path/to/the/changed/code/directory(ies)
```
to check if there is any other issues to fix.

## Downloading pretrained models

You can download MobileNetV3 model with the following script:
```shell
cd models
sh download_pretrained_base.sh
```
The file will be downloaded into ``models``.

## Validation videos

If you want to download the 3 test videos on the 3 portions of the Auterrive riverbank, run:

```
cd data
sh download_validation_videos.sh
```

This will download the 3 videos in distinct folders of ``data/validation_videos``.

## Serving

### Development
Setting up the server and testing: from ``surfnet/`` directory, you may run a local flask developement server with the following command:

```shell
export FLASK_APP=src/plasticorigins/serving/app.py
poetry run flask run
```

### Production
Setting up the server and testing: from ``surfnet/`` directory, you may run a local wsgi gunicorn production server with the following command:

```shell
PYTHONPATH=./src gunicorn -w 5 --threads 2 --bind 0.0.0.0:8001 --chdir ./src/serving/ wsgi:app
```

### Test surfnet API
Then, in order to test your local dev server, you may run:
```shell
curl -X POST http://127.0.0.1:5000/ -F 'file=@/path/to/video.mp4' # flask
```
Change port ``5000`` to ``8001`` to test on gunicorn or ``8000`` to test with Docker and gunicorn.

### Docker
You can build and run the surfnet AI API within a Docker container.

Docker Build:
```shell
docker build -t surfnet/surfnet:latest .
```

Docker Run:
```shell
docker run --env PYTHONPATH=/src -p 8000:8000 --name surfnetapi surfnet/surfnet:latest
```

### Makefile
You can use the makefile for convenience purpose to launch the surfnet API:
```shell
make surfnet-dev-local # with flask
make surfnet-prod-local # with gunicorn
make surfnet-prod-build-docker # docker build
make surfnet-prod-run-docker # docker run
```

### Kubernetes
To ease production operation, the surfnet API can be deployed on top of kubernetes (k8s) cluster. A pre-built Docker image is available on ghcr.io to be deployed using the surfnet.yaml k8s deployment file. To do so, change directory to ``k8s/``, then once you are connected to your k8s cluster simply enter:
```shell
kubectl apply -y surfnet.yaml
```

- *Remark*: we use a specific surfnet k8s node pool label for our Azure production environment on aks. If you want to test deployment on a default k8s cluster using system nodes, you have either to use default surfnet.yaml file or remove the nodeSelector section from others deployment files (aks, gke).

After the deployment is done, create a service to expose the surfnet API to be publicly accessible over the Internet.
```shell
kubectl expose deployment surfnet --type=LoadBalancer --name=surfnet-api
kubectl get service surfnet-api
```