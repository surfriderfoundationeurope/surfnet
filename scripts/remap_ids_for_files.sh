files_dir=experiments/tracking/our_tracker_with_fairmot_detections
cd $files_dir
for f in *.txt; do 
    cd ~/repos/surfnet
    python src/remap_ids.py --input_file ${files_dir}/$f --min_len_tracklet 1 --output_name $f
done