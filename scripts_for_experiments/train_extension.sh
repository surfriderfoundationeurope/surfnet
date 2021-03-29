. scripts_for_experiments/shell_variables.sh 

model_name='surfnet32'
sigma2='2'
alpha='2'
beta='4'
lr='1e-5'

experiment_name=${model_name}'_alpha_'${alpha}'_beta_'${beta}'_lr_'${lr}

output_dir='experiments/extension/'${experiment_name}
create_clean_directory $output_dir 

trap "exit" INT TERM 
trap "kill 0" EXIT

tensorboard --logdir=${output_dir} & 

python train_extension.py \
    --model ${model_name} \
    --batch-size 16 \
    --data-path ${BASE_NETWORK_HEATMAPS} \
    --log-dir ${output_dir} \
    --output-dir ${output_dir} \
    --alpha ${alpha} \
    --lr ${lr} \
    --beta ${beta} && fg
wait

