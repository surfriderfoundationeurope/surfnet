rm -f shell_variables.sh
cat > shell_variables.sh <<EOF
create_clean_directory(){
    dir_name=\$1
    if [ -d "\$dir_name" ]; then
        echo "Removing \$dir_name"
        rm -rf "\$dir_name"
    elif [ -f "\$dir_name" ]; then
        echo "File with this name already exists, not a directory."
        exit
    fi
    if mkdir "\$dir_name"; then
        echo "Clean directory created: \$dir_name"
        return 0
    else
        echo "Creating directory failed: \$dir_name"
        return 1
    fi 
}

export CUDA_VISIBLE_DEVICES=0
export IMAGES='./data/images'
export BASE_NETWORK_HEATMAPS='./data/extracted_heatmaps'
export BASE_PRETRAINED_WEIGHTS='./external_pretrained_networks/centernet_pretrained.pth'
export SYNTHETIC_VIDEOS='./data/generated_videos'
export DOWNSAMPLING_FACTOR='4'
export SYNTHETIC_OBJECTS='./data/synthetic_objects'
export VALIDATION_VIDEOS='./data/validation_videos'
export EXTERNAL_DETECTIONS='./data/detector_results'
EOF


