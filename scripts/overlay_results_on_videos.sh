cwd=$(pwd)
sequences=short_segments_12fps
tracker_name=ours_EKF_1_12fps_v2_7_tau_5
videos_dir=data/validation_videos/all/${sequences}/videos
mot_gt_files_dir=data/validation_videos/all/${sequences}/mot_gt_files/surfrider-test
results_dir=external/TrackEval/data/trackers/surfrider_${sequences}/surfrider-test/${tracker_name}/data
cd ${results_dir} 

for f in *; 
    do
        base_name="${f%.*}"
        cd ${cwd}
        python src/overlay_tracking_results_on_video.py \
            --input_video ${videos_dir}/${base_name}.mp4 \
            --input_gt_mot_file ${mot_gt_files_dir}/${base_name}/gt/gt.txt \
            --input_mot_file ${results_dir}/${base_name}.txt \
            --write True \
            --output_name overlay_${tracker_name}_${base_name}
    done
