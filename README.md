# Installation 

## General requirements

Follow these steps in that order exactly:
```shell
git clone --recursive https://github.com/mchagneux/surfnet.git <folder-for-surfnet> -b release
conda create -n surfnet pytorch=1.7 torchvision=0.8.1 -c pytorch 
conda activate surfnet
cd <folder-for-surfnet>
pip install -r requirements.txt

cd ..
git clone git@github.com:pykalman/pykalman.git <folder-for-pykalman>
cd <folder-for-pykalman> 
python setup.py install

cd scripts 
sh init_shell_variables.sh

cd ../src/detection/centernet/networks/DCNv2
sh make.sh
```
## Validation 

Add your videos to [data/validation_videos](data/validation_videos) or download the ones from the paper with the script in the folder. Then: 

```shell
sh script/tracks.sh
```

The tracking and count results will be in [experiments/tracking](experiments/tracking) in the subfolder of your experiment (default="test").

If you want to overlay the tracks on your original, run: 

```shell 
python src/overlay_tracking_results_on_video.py \
    --input_video <path-to-video> \
    --input_mot_file <path-to-tracking-results-for-video> \
    --write True 
    --output_name <name-of-the-output-file>
```

