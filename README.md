# Automated object counting on riverbanks

## Project Branches:

### release:

This is the main/production branch (DO NOT PUSH DIRECTLY TO RELEASE)
### dev: 

This is the developpement branch (DO NOT PUSH DIRECTLY TO DEV)
### Feature Branches

These are the branches where you can develop new features for the project. In order to create a feature branch:

- Make sure that your **dev** branch is up to date
    ```shell
    git checkout dev
    git pull dev
    ```
- Create a new branch from **dev** with the name **feature/name_of_your_feature**
    ```shell
    git checkout -b feature/name_of_your_feature
    ```
- Once your feature developpement is complete, make a Pull Request of your feature branch to **dev**
### Research Branches: 

These are branches made for research purposes and they are named **research/name_of_your_subject**
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
Here we use python version 3.9
```shell
poetry env use 3.9
```

### Install dependencies
```shell
poetry install
```

### Code Linting and Formatting:

pre-commits have been added to format and check the linting of the code before any commit. This process will run:
- PyUpgrade: to make sure that the code syntax is up to date with the latest python versions
- Black: which is a code formatter 
- Flake8: to check that the code is properly formatted.

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
export FLASK_APP=src/plasticorigins/serving/app.py
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

### Prerelease: (Pull Request to Dev branch)
#### Check or Bump version:

Check the current version of the product:

Docker Build:
```shell
poetry version
```

Bump the version to the product:

```shell
poetry version <bump-rule>
```
bump rules can be found in : https://python-poetry.org/docs/cli/#:~:text=with%20concrete%20examples.-,RULE,-BEFORE
**choose carefully the one that corresponds to your bump: for the prerelease we will use:**
- **prepatch**
- **preminor**
- **premajor**

make sure that in you pyproject.toml your version ends with **-alpha.0**

#### Publish the prerelease to pypi:

In order to publish your prerelease to PyPi, all you need to do is open a Pull Request of your current branch to **Dev** branch. Once the PR is approved and merged, the Prerelease will be done automatically with a github workflow.

### Release: (Pull Request to Release branch):
In order to publish a release version to PyPi, all you have to do is open a Pull Request of the **Dev** branch into the **Release** branch. Once the PR is approved and merged, the Release will be done automatically with a github workflow.

## Testing:
To launch the tests you can run this command
```shell
poetry run coverage run -m pytest -s && poetry run coverage report -m
```

## Mkdocs Documentation:
You need to install the following packages:
```shell
pip install mkdocs
pip install mkdocstrings
```
To run the mkdocs documentation, you can run the following lines below:
```shell
cd src
mkdocs serve
```
The documentation will be serving on http://127.0.0.1:8000/.

## Configuration

`src/serving/inference.py` contains a Configuration dictionary that you may change:
- `skip_frames` : `3` number of frames to skip. Increase to make the process faster and less accurate.
- `kappa`: `7` the moving average window. `1` prevents the average, avoid `2` which is ill-defined.
- `tau`: `4` the number of consecutive observations necessary to keep a track. If you increase `skip_frames`, you should lower `tau`.

## Datasets and Training

Consider other branches for that!
