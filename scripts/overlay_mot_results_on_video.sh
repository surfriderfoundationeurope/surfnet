sequences=short_segments_12fps
sequence_name=part_1_segment_7
tracker_name=sort
input_gt_mot_file=external/TrackEval/data/gt/surfrider_$sequences/surfrider-test/$sequence_name/gt/gt.txt
input_video=data/validation_videos/$sequences/videos/$sequence_name.mp4
input_mot_file=external/TrackEval/data/trackers/surfrider_$sequences/surfrider-test/$tracker_name/data/$sequence_name.txt
output_name=${sequences}_${sequence_name}_${tracker_name}

python src/overlay_tracking_results_on_video.py \
    --input_video ${input_video} \
    --input_mot_file ${input_mot_file} \
    --output_name ${output_name} \
    --write True 
    # --input_gt_mot_file ${input_gt_mot_file}
