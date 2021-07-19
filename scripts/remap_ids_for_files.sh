fps=6
segments=long

for tau in 1 2 3 4 5; do 
    files_dir=experiments/tracking/${segments}_segments/${fps}fps_tau_0
    output_dir=experiments/tracking/${segments}_segments/${fps}fps_tau_${tau}
    mkdir ${output_dir}

    cd ${files_dir}

    for f in *.txt; do 
        cd ~/repos/surfnet
        python src/remap_ids.py --input_file ${files_dir}/$f --min_len_tracklet ${tau} --output_name ${output_dir}/$f
    done
done 