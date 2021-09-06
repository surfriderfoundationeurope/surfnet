cwd=$(pwd)
segments=short
algorithm=UKF
original_dir=experiments/tracking/${algorithm}/${segments}_segments
fps=12
target_dir=${cwd}/external/TrackEval/data/trackers/surfrider_${segments}_segments_${fps}fps/surfrider-test
cd ${original_dir}

for dir in ${fps}fps*; do 
    tracker_dir=${target_dir}/ours_${algorithm}_${dir}
    mkdir ${tracker_dir}
    mkdir ${tracker_dir}/data
    cp ${dir}/* ${tracker_dir}/data
done



