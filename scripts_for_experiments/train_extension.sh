. scripts_for_experiments/shell_variables.sh 

model_name='surfnet32'
sigma2='2'
alpha='2'
beta='4'
lr='1e-5'

experiment_name=${model_name}'_sigma2_'${sigma2}'_alpha_'${alpha}'_beta_'${beta}

output_dir='experiments/extension/'${experiment_name}
echo ${output_dir}
create_clean_directory $output_dir 

trap "exit" INT TERM 
trap "kill 0" EXIT

tensorboard --logdir=${output_dir} & 

python train_extension.py \
    --model ${model_name} \
    --batch-size 8 \
    --data-path ${BASE_EXTRACTED_HEATMAPS} \
    --log-dir ${output_dir} \
    --output-dir ${output_dir} \
    --sigma2 ${sigma2} \
    --alpha ${alpha} \
    --lr ${lr} \
    --beta ${beta} && fg

wait

