. scripts_for_experiments/shell_variables.sh 

model_name='surfnet32'
sigma2='2'
alpha='2'
beta='4'
lr='1e-5'
detail='single_class_video_frames'

experiment_name=${model_name}'_alpha_'${alpha}'_beta_'${beta}'_lr_'${lr}'_'${detail}

output_dir='experiments/extension/'${experiment_name}
create_clean_directory $output_dir 

base_name='dla_34_downsample_4_alpha_2_beta_4_lr_6.25e-5_single_class_video_frames'
annotations_dir='data/synthetic_videos_dataset/annotations'
trap "exit" INT TERM 
trap "kill 0" EXIT

tensorboard --logdir=${output_dir} & 

python train_extension.py \
    --model ${model_name} \
    --batch-size 16 \
    --heatmaps_dir ${BASE_NETWORK_HEATMAPS} \
    --base_name ${base_name} \
    --annotations_dir ${annotations_dir} \
    --log-dir ${output_dir} \
    --output-dir ${output_dir} \
    --alpha ${alpha} \
    --lr ${lr} \
    --beta ${beta} && fg
wait

