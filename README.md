# Automated object counting on riverbanks

## Release Branch - Installation

Follow these steps in that order exactly:

### Clone the project
```shell
git clone https://github.com/surfriderfoundationeurope/surfnet.git <folder-for-surfnet> -b release
cd <folder-for-surfnet>
```
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
## Downloading pretrained models

You can download MobileNetV3 model with the following script:
```shell
cd models
sh download_pretrained_base.sh
```
The file will be downloaded into [models](models).

## Validation videos

If you want to download the 3 test videos on the 3 portions of the Auterrive riverbank, run:

```
cd data
sh download_validation_videos.sh
```

This will download the 3 videos in distinct folders of [data/validation_videos](data/validation_videos).

## Serving

### Development
Setting up the server and testing: from surfnet/ directory, you may run a local flask developement server with the following command:

```shell
export FLASK_APP=src/serving/app.py
poetry run flask run
```

### Production
Setting up the server and testing: from surfnet/ directory, you may run a local wsgi gunicorn production server with the following command:

```shell
PYTHONPATH=./src gunicorn -w 5 --threads 2 --bind 0.0.0.0:8001 --chdir ./src/serving/ wsgi:app
```

### Test surfnet API
Then, in order to test your local dev server, you may run:
```shell
curl -X POST http://127.0.0.1:5000/ -F 'file=@/path/to/video.mp4' # flask
```
Change port 5000 to 8001 to test on gunicorn or 8000 to test with Docker and gunicorn.

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
To ease production operation, the surfnet API can be deployed on top of kubernetes (k8s) cluster. A pre-built Docker image is available on ghcr.io to be deployed using the surfnet.yaml k8s deployment file. To do so, change directory to k8s/, then once you are connected to your k8s cluster simply enter:
```shell
kubectl apply -y surfnet.yaml
```
Remark: we use a specific surfnet k8s node pool label for our Azure production environment on aks. If you want to test deployment on a default k8s cluster using system nodes, you have either to use default surfnet.yaml file or remove the nodeSelector section from others deployment files (aks, gke).

After the deployment is done, create a service to expose the surfnet API to be publicly accessible over the Internet.
```shell
kubectl expose deployment surfnet --type=LoadBalancer --name=surfnet-api
kubectl get service surfnet-api
```

## Release plasticorigins to pypi:

### Check or Bump version:

Check the current version of the product:

```shell
poetry version
```

Bump the version to the product:

```shell
poetry version <bump-rule>
```
bump rules can be found in : https://python-poetry.org/docs/cli/#:~:text=with%20concrete%20examples.-,RULE,-BEFORE
**choose carefully the one that corresponds to your bump (we usally will be using "patch" as a bump-rule**
### Build the project
```shell
poetry build
```

### Publish the project to pypi:

```shell
poetry publish --username your_pypi_username --password your_pypi_password
```

## Testing:
To launch the tests you can run this command
```shell
poetry run coverage run -m pytest -s && poetry run coverage report -m
```
## Configuration

`src/serving/inference.py` contains a Configuration dictionary that you may change:
- `skip_frames` : `3` number of frames to skip. Increase to make the process faster and less accurate.
- `kappa`: `7` the moving average window. `1` prevents the average, avoid `2` which is ill-defined.
- `tau`: `4` the number of consecutive observations necessary to keep a track. If you increase `skip_frames`, you should lower `tau`.

## Datasets and Training

Consider other branches for that!
