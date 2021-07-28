# Installation 

## General requirements

```shell
python3 -m venv <folder-for-your-clean-environment>
source  <folder-for-your-environment>/bin/activate
pip install -r requirements.txt

git clone git@github.com:pykalman/pykalman.git <folder-for-pykalman>
cd <folder-for-pykalman> 
python setup.py install
```

## DCNv2 dependencies 

```shell 
cd src/base/centernet/networks
rm -rf DCNv2
git clone https://github.com/jinfagang/DCNv2_latest.git
mv DCNv2_latest/ DCNv2/ 
cd DCNv2
./make.sh
```


## Data
You can download some data using the following scripts:

```shell
cd data
sh download_surfrider_images.sh
sh download_validation_videos.sh
```

Then run the following scripts to initialize the correct paths: 

```shell
cd scripts
sh init_shell_variables.sh
```
The newly created file allows you to modify the filepaths to your convenience.

# Experiments

All experiment scripts are found in [scripts/](scripts/)  under explicit names.

## Detection network

### Training 


```shell
sh scripts/train_detector.sh
```
Then open Tensorboard in your browser to follow training. 

### Obtaining heatmaps from the base network 
If you need a pretrained version of the base network you can run: 

```shell 
cd external_pretrained_models
sh download_pretrained_base.sh 
```
This will download one of our pretrained models. 

You can modify the BASE_PRETRAINED variable in scripts_for_experiments/shell_variables.sh to point to the base networks weights that you obtained yourself through retraining, for example.
