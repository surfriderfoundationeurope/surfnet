. scripts_for_experiments/shell_variables.sh

python extract_and_save_heatmaps.py \
	--video-dir ${SYNTHETIC_VIDEOS} \
	--output-dir ${BASE_NETWORK_HEATMAPS} \
	--weights  ${BASE_PRETRAINED_WEIGHTS} \
	--downsampling-factor ${DOWNSAMPLING_FACTOR} \
	--model 'dla_34' \
	--my-repo 