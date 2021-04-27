. scripts_for_experiments/shell_variables.sh


rm -r ${BASE_NETWORK_HEATMAPS}/*

python extract_and_save_heatmaps.py \
	--dataset_dir ${SYNTHETIC_VIDEOS_DATASET} \
	--output_dir ${BASE_NETWORK_HEATMAPS} \
	--weights  ${BASE_PRETRAINED} \
	--downsampling_factor ${DOWNSAMPLING_FACTOR}