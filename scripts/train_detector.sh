. scripts/shell_variables.sh

downsampling_factor='4'
alpha='2'
beta='4'
lr=1.25e-4
lr_step=140
model_name='mobilenetv3small'
dataset='surfrider'
batch_size=16
experiment_name='full_classes_cpu'
num_classes=9

output_dir='experiments/detection/'${experiment_name}
create_clean_directory $output_dir


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
    --device cpu \
    --epochs 5 \
    --batch-size 4 \
    --workers 4 \
    --num-classes ${num_classes}\
    --lr_step ${lr_step}
