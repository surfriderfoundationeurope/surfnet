# Automated object counting on riverbanks

## Installation 

Follow these steps in that order exactly:
```shell
git clone https://github.com/mchagneux/surfnet.git <folder-for-surfnet> -b release
conda create -n surfnet pytorch torchvision -c pytorch 
conda activate surfnet
cd <folder-for-surfnet>
pip install -r requirements.txt

cd ..
git clone git@github.com:pykalman/pykalman.git <folder-for-pykalman>
cd <folder-for-pykalman> 
python setup.py install

cd scripts 
sh init_shell_variables.sh
```
## Downloading pretrained models

You can download three pretrained models with the following script:
```shell 
cd models 
sh download_pretrained_base.sh
```
There is DLA_34 with deformable convolutions, ResNet18 without deformable convolutions and MobileNetV3 (small version, from Torchvision) without deformable convolutions.
The files will be downloaded into  [models](models).


## Downloading Surfrider datasets 

### Image dataset 

If you want to download a small portion of the Surfrider images, do: 

```shell 
cd data
sh download_small_dataset.sh
```
This will download 500 images in [data/images/images](data/images/images) and the associated annotations in [data/images/annotations/instances.json](data/images/annotations/instances.json).

Then the following file to split the dataset into train and test:
```
python src/datasets/coco_split_train_test.py
```
If you want to visualize the images, run: 


```
python src/datasets/visualize_coco_boxes.py
```
---
*Warning: the remaining section will download more that 5GB of images.*

If you want to download the rest, you need to install AzCopy (see https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10 for example). You also need to personally ask me the URL and SAS token (temporary workaround). When this is done, simply run: 

```shell 
cd data/images
azcopy copy --recursive '<URL+SAS>' './'
mv images2label/* images/
rm -rf images2label
```

This will add the remaining images into the previsouly created folder.

Finally, run: 

```shell 
python src/datasets/surfrider_db_to_coco_converter.py
python src/datasets/merge_coco_annotations.py
python src/datasets/coco_split_train_test.py
```
This will download the remaining annotations, merge them with the previous ones, and re-split into train and test. 

### Validation videos 

If you want to downlaod the 3 test videos on the 3 portions of the Auterrive riverbank, run: 

```
cd data 
sh download_validation_videos.sh
```

This will download the 3 videos in distinct folders of [data/validation_videos](data/validation_videos).


## Run 

If you have custom videos, add them to [data/validation_videos](data/validation_videos) in a new subfolder. Then: 

```shell
sh scripts/track.sh
```

By default, this runs tracking on the first riverbank sequence (T1). You can change to the `--arch` parameter to:
* `mobilenet_v3_small`
* `res_18`
* `dla_34`

The default harware used is the CPU, but you can change the `--device` parameter to `cuda` and PyTorch will automatically select a GPU if there is one. In this case you should set a higher `--detection_batch_size` to improve detection speed. You can also add `--preload_frames` if you want all video frames to be loaded into the RAM before detections and tracking.




The tracking and count results will be in [experiments/tracking](experiments/tracking) in the subfolder of your experiment (default="test").

If you want to overlay the tracks on the video, run: 

```shell 
python src/overlay_tracking_results_on_video.py \
    --input_video <path-to-video> \
    --input_mot_file <path-to-tracking-results-for-video> \
    --write True \
    --output_name <name-of-the-output-file> \
    --skip_frames <nb-of-skipped-frames-when-tracking>
```

Note that by default we set `skip_frames = 3` in [scripts/track.sh](scripts/track.sh) so that everything is read at 6fps. You must use the same number when generating the overlay.
If you set `--write False` the overlay will be directly displayed without saving to a file and you must use the keyboard to step into the next frame.
