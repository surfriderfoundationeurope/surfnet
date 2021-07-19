segments=long
original_dir=experiments/tracking/${segments}_segments
fps=6
target_dir=/home/infres/chagneux/repos/TrackEval/data/trackers/surfrider_${segments}_segments_${fps}fps/surfrider-test
cd ${original_dir}

for dir in ${fps}fps*; do 
    tracker_dir=${target_dir}/ours_${dir}
    mkdir ${tracker_dir}
    mkdir ${tracker_dir}/data
    cp ${dir}/* ${tracker_dir}/data
done



