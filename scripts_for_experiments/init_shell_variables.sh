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

eval "\$(conda shell.bash hook)"
conda activate surfnet_DLA

export CUDA_VISIBLE_DEVICES=
export IMAGES_PATH=''
export BASE_EXTRACTED_HEATMAPS=''
export BASE_PRETRAINED=''
export SYNTHETIC_VIDEOS_PATH=''
export DOWNSAMPLING_FACTOR=''
EOF

