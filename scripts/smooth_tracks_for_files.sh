fps=12
segments=short
algorithm='EKF_1'
frames_dir=data/external_detections/FairMOT/surfrider_${segments}_segments_${fps}fps
files_dir=experiments/tracking/${algorithm}/${segments}_segments/${fps}fps_tau_0

# mkdir experiments/tracking/${algorithm}_smoothed
mkdir experiments/tracking/${algorithm}_smoothed/${segments}_segments
output_dir=experiments/tracking/${algorithm}_smoothed/${segments}_segments/${fps}fps_v0_tau_0
mkdir ${output_dir}

cd ${files_dir}

for f in *.txt; do 
    base_name="${f%.*}"
    cd ~/repos/surfnet
    python src/smooth_tracks.py \
        --input_file ${files_dir}/$f \
        --frames_file ${frames_dir}/${base_name}/saved_frames.pickle \
        --output_name ${output_dir}/$f \
        --confidence_threshold 0.5 \
        --downsampling_factor 1
done
