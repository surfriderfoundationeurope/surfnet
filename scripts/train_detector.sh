. scripts/shell_variables.sh 

downsampling_factor=4
alpha=2
beta=4
lr=1.25e-4
lr_step=140
model_name='mobilenetv3small'
dataset='surfrider'
batch_size=16
experiment_name='3500_images_mobilenet'

output_dir='experiments/detection/'${experiment_name}
create_clean_directory $output_dir 

# trap "exit" INT TERM 
# trap "kill 0" EXIT

# tensorboard --logdir=${output_dir} & 

python src/train_detector.py \
    --model ${model_name} \
    --dataset ${dataset} \
    --data-path ${IMAGES} \
    --batch-size ${batch_size} \
    --output-dir ${output_dir} \
    --logdir ${output_dir} \
    --downsampling-factor ${DOWNSAMPLING_FACTOR} \
    --alpha ${alpha} \
    --beta ${beta} \
    --lr ${lr} \
    --epochs 290 \
    --lr_step ${lr_step} &> ${output_dir}/logs.out &


