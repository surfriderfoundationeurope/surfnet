This site contains the project documentation for the
`plasticorigins` package used for trash object detection.

## Table Of Contents

1. [Tutorials](tutorials.md)
2. [Installation](installation.md)
3. [How-To Guides](how-to-guides.md)


## Project Overview

::: plasticorigins

## Acknowledgements


## Project layout

    mkdocs.yml    # The configuration file.

    docs/
        detection/
            coco_utils.md
            detect.md
            losses.md
            models.md
            transforms.md
            yolo.md
        serving/
            inference.md
        tools/
            files.md
            misc.md
            optical_flow.md
            video_readers.md
        tracking/
            postprocess_and_count_tracks.md
            track_video.md
            trackers.md
            utils.md
        training/
            data_processing.md
            make_dataset.md
        how-to-guides.md  # The page to guide users through the package.
        index.md  # The documentation homepage.
        installation.md   # The installation page.
        tutorials.md   # The tutorials page.
        
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

