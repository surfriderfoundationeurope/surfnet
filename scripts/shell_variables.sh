create_clean_directory(){
    dir_name=$1
    if [ -d "$dir_name" ]; then
        echo "Removing $dir_name"
        rm -rf "$dir_name"
    elif [ -f "$dir_name" ]; then
        echo "File with this name already exists, not a directory."
        exit
    fi
    if mkdir "$dir_name"; then
        echo "Clean directory created: $dir_name"
        return 0
    else
        echo "Creating directory failed: $dir_name"
        return 1
    fi 
}

# eval "$(conda shell.bash hook)"
# conda activate surfnet_DLA

export CUDA_VISIBLE_DEVICES=0
export IMAGES='./data/surfrider_images/'
export BASE_NETWORK_HEATMAPS='./data/extracted_heatmaps/'
export BASE_PRETRAINED='./external_pretrained_models/base_single_class_video_frames.pth'
export SYNTHETIC_VIDEOS_DATASET='./data/synthetic_videos_dataset/'
export DOWNSAMPLING_FACTOR='4'
export SYNTHETIC_OBJECTS='./data/synthetic_objects/'
export ORIGINAL_VIDEOS=''
export EXTENSION_PRETRAINED='experiments/extension/surfnet32_alpha_2_beta_4_lr_1e-5_single_class_video_frames/model_72.pth'


