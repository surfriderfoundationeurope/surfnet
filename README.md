# Installation 

## General requirements

Follow these steps in that order exactly:
```shell
conda create -n surfnet pytorch=1.7 torchvision=0.8.1 -c pytorch 
conda activate surfnet
pip install -r requirements.txt

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



