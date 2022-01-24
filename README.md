# Automated object counting on riverbanks

## TODO
- **Final evaluations and hyperparameter selection**:
  - [ ] Step 0: redefine a proper test split for the image dataset where all images come from environments that are not seen during training. Currently, a random portion of all images is selected for testing and therefore some of the images depict objects already seen at training (e.g from a different angle), etc. Some images also correspond to scenes that are in the test videos (because volonteers also took pictures aside from the footage). Based on dates a practical solution is to:
    - Remove all images that belong to the videos by finding the date of the corresponding expeditions in Auterrive
    - Select test images by restricting them to one expedition that is not seen for training (using dates)
  - [ ] Step 1: check the evaluation code, e.g.:
    - Preprocessing of the images in the test split (does the 'ValTransforms' class follow best practises for testing ?)
    - Evaluation procedure at a given threshold: Hungarian algorithm + definition of TP/FP/FN (and definition of the threshold for accepting TP, small problem with image size and depth of view)
    - Plotting of PR/F1 and distance to positives
  - [ ] Step 2: evaluate all models (DLA34, ResNet18, MobileNetv3)
    - Find best threshold for each model given PR curves
    - Choose a final model
  - [ ] Step 3: if optimal detection threshold is significantly different from what was used in paper, calibrate the tracking/counting threshold accordingly.

- **Multiclass version**:
  - [x] Gather all newly labeled images and compute statistics then take decisions
    - Proportion of images per class
  - [ ] Decide of a strategy to mitigate imbalance: should some classes be merged/discarded/added ?
  - [ ] Train one of the models with multiclass output
  - [ ] Quantify performance (for ex aggregate multiclass results to single class after forward pass and compare with single output network)

## Installation

## Dev Branch - Installation


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


## Datasets and Training

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

### more images

*Warning: the remaining section will download more that 5GB of images.*

If you want to download the rest, you need to install AzCopy (see https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10 for example). You also need to personally ask me the URL and SAS token (temporary workaround). When this is done, simply run:

```shell
mkdir -p data/images
cd data/images
azcopy copy --recursive '<URL+SAS>' './'
mv images2label/ images
```

This will add the remaining images into the previsouly created folder.

Finally, run:

```shell
python src/datasets/surfrider_db_to_coco_converter.py
mv instances_multiclass.json instances.json
python src/datasets/coco_split_train_test.py
```

if you have downloaded the small dataset, replace the `mv` command by a merge command: `python src/datasets/merge_coco_annotations.py`

This will download the remaining annotations, merge them with the previous ones, and re-split into train and test.

### training

see `train_detector.py` to run training.

## Serving (dev mode)

From the main directory, you may run a local flask test server with the following command:

```shell
export FLASK_APP=src/serving/app.py
flask run
```

Then, in order to test your local server, you may run:
```shell
curl -X POST http://127.0.0.1:5000/ -F 'file=@/path/to/video.mp4'
```
