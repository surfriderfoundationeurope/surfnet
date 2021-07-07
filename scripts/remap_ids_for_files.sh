files_dir=experiments/tracking/all_short_segments_count_thres_0
cd $files_dir
for f in *.txt; do 
    cd ~/repos/surfnet
    python src/remap_ids.py --input_file ${files_dir}/$f --min_len_tracklet 8 --output_name $f
done