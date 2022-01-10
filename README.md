# Automated object counting on riverbanks

## Release Branch - Installation

Follow these steps in that order exactly:
```shell
git clone https://github.com/mchagneux/surfnet.git <folder-for-surfnet> -b release
conda create -n surfnet pytorch torchvision -c pytorch
conda activate surfnet
cd <folder-for-surfnet>
pip install -r requirements.txt

cd scripts
sh init_shell_variables.sh
cd ..
```
## Downloading pretrained models

You can download MobileNetV3 model with the following script:
```shell
cd models
sh download_pretrained_base.sh
cd ..
```
The file will be downloaded into [models](models).

## Validation videos

If you want to downlaod the 3 test videos on the 3 portions of the Auterrive riverbank, run:

```
cd data
sh download_validation_videos.sh
cd ..
```

This will download the 3 videos in distinct folders of [data/validation_videos](data/validation_videos).

## Serving

Setting up the server and testing; from the main directory, you may run a local flask test server with the following command:

```shell
export FLASK_APP=src/serving/app.py
flask run
```

Then, in order to test your local server, you may run:
```shell
curl -X POST http://127.0.0.1:5000/ -F 'file=@/path/to/video.mp4'
```

## Datasets and Training

Consider other branches for that!
