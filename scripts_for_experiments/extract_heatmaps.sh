. scripts_for_experiments/shell_variables.sh

python extract_and_save_heatmaps.py \
	--input-dir ${SYNTHETIC_VIDEOS_PATH} \
	--output-dir ${BASE_NETWORK_HEATMAPS} \
	--weights  ${BASE_PRETRAINED} \
	--downsampling-factor ${DOWNSAMPLING_FACTOR} \
	--model 'dla_34' \
	--my-repo \
	--from-videos \
	--extract-flow