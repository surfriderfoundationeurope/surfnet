export CUDA_VISIBLE_DEVICES=2
dataset_dir='/home/infres/chagneux/repos/surfnet/data/validation_videos/all/long_segments/videos'
cd ${dataset_dir}
output_dir_name=surfrider_long_segments
for f in *.mp4; do
    echo $f
    cd ~/repos/FairMOT
    base_name="${f%.*}"
    echo $base_name
    dir_for_video=$output_dir_name/${base_name}

    python src/demo.py mot \
        --load_model exp/mot/surfrider_1070_images_290_epochs/model_290.pth \
        --conf_thres 0.4 \
        --input-video ${dataset_dir}/$f \
        --output-root ./${dir_for_video} \
        --not_reg_offset
    clean=0
    python remap_ids.py --input_file  ./${dir_for_video}/results.txt --min_len_tracklet $clean --output_name $f
    mv $base_name.txt ${dir_for_video}/results_clean_$clean.txt

    clean=1
    python remap_ids.py --input_file  ./${dir_for_video}/results.txt --min_len_tracklet $clean --output_name $f
    mv $base_name.txt ${dir_for_video}/results_clean_$clean.txt

    rm ./${dir_for_video}/results.txt
    rm -rf ${dir_for_video}/frame
    mv saved_detections.pickle ${dir_for_video}/saved_detections.pickle
    mv saved_frames.pickle ${dir_for_video}/saved_frames.pickle

done

