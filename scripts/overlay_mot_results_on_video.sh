sequences=long_segments_12fps
sequence_name=part_1_1
tracker_name=ours_EKF_order_1_hungarian_12fps_tau_6
input_gt_mot_file=../TrackEval/data/gt/surfrider_$sequences/surfrider-test/$sequence_name/gt/gt.txt
input_video=data/validation_videos/all/$sequences/videos/$sequence_name.mp4
input_mot_file=../TrackEval/data/trackers/surfrider_$sequences/surfrider-test/$tracker_name/data/$sequence_name.txt
output_name=${sequences}_${sequence_name}_${tracker_name}

python src/overlay_tracking_results_on_video.py \
    --input_video ${input_video} \
    --input_mot_file ${input_mot_file} \
    --output_name ${output_name} \
    --write True \
    --input_gt_mot_file ${input_gt_mot_file}
