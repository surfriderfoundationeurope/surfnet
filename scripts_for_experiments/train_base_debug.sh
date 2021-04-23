. scripts_for_experiments/shell_variables.sh 

downsampling_factor='4'
alpha='2'
beta='4'
lr=6.25e-5
model_name='dla_34'
detail='_3'
experiment_name=${model_name}'_downsample_'${downsampling_factor}'_alpha_'${alpha}'_beta_'${beta}'_lr_'${lr}${detail}

output_dir='experiments/base/'${experiment_name}
create_clean_directory $output_dir 

trap "exit" INT TERM 
trap "kill 0" EXIT

tensorboard --logdir=${output_dir} & 

python -m debugpy --listen 5678 --wait-for-client train_base.py \
    --model ${model_name} \
    --data-path ${IMAGES} \
    --batch-size 2 \
    --output-dir ${output_dir} \
    --logdir ${output_dir} \
    --downsampling-factor ${DOWNSAMPLING_FACTOR} \
    --alpha ${alpha} \
    --beta ${beta} \
    --lr ${lr} && fg
wait    



