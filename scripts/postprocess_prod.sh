. scripts/shell_variables.sh

experiment_name='prod_'$1
output_dir=experiments/tracking/${experiment_name}

filename=`ls $output_dir| head -n 1`
path=${filename%/*}
file=${f##*/}
base=${file%%.*}
ext=${file#*.}
output_file=${path}/${base}_f.json

python src/postprocess_and_count_tracks.py \
    --input_file $ffilename\
    --kappa 7  \
    --tau 5 \
    --output_name $output_file
