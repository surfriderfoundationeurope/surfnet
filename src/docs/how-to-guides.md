# How to guides 

## Project Layout

    plasticorigins/
            detection/
                centernet/
                    networks/
                        mobilenet.py
                    models.py
                coco_utils.py
                detect.py
                losses.py
                models.py
                transforms.py
                yolo.py
            serving/
                app.py
                config.py
                inference.py
                wsgi.py
            tools/
                files.py
                misc.py
                optical_flow.py
                video_readers.py
            tracking/
                postprocess_and_count_tracks.py
                track_video.py
                trackers.py
                utils.py
            training/
                data/
                    data_processing.py
                    make_dataset.py
                visualization/
                    categories.py

## Modules

### Detection module

Module of the inference part for the object detection phase. It includes several functions for pre-processing and image transformations, as well as the creation of Deep Learning models for detection (formatting) and evaluation of detections with loss functions. 

### Serving module

Module of the inference part to manage the server connection part launching a Flask web application.

### Tools module

Useful function module for heatmaps calculations, model predictions, playback and conversion of input videos and loading of external data.

### Tracking module

Module of the inference part dedicated to tracking waste on videos/ images.

### Training module

Module of the training part allowing the construction of annotations and label files used for the training phases.