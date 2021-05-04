. scripts_for_experiments/shell_variables.sh 

downsampling_factor='4'
alpha='2'
beta='4'
lr=6.25e-5
model_name='dla_34'
detail='_single_class_video_frames_continued'
dataset='surfrider_video_frames'
experiment_name=${model_name}'_downsample_'${downsampling_factor}'_alpha_'${alpha}'_beta_'${beta}'_lr_'${lr}${detail}

output_dir='experiments/base/'${experiment_name}
create_clean_directory $output_dir 

trap "exit" INT TERM
trap "kill 0" EXIT

tensorboard --logdir=${output_dir} & 

python train_base.py \
    --model ${model_name} \
    --dataset ${dataset} \
    --data-path ${SYNTHETIC_VIDEOS_DATASET} \
    --batch-size 16 \
    --output-dir ${output_dir} \
    --logdir ${output_dir} \
    --downsampling-factor ${DOWNSAMPLING_FACTOR} \
    --alpha ${alpha} \
    --beta ${beta} \
    --lr ${lr} \
    --resume '/home/infres/chagneux/repos/surfnet/experiments/base/dla_34_downsample_4_alpha_2_beta_4_lr_6.25e-5_single_class_video_frames/model_69.pth' && fg
wait    



