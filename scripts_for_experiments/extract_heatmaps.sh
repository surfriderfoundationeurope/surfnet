. scripts_for_experiments/shell_variables.sh

python -m debugpy --listen 5678 --wait-for-client extract_and_save_heatmaps.py \
	--video-dir ${SYNTHETIC_VIDEOS_PATH} \
	--output-dir ${BASE_EXTRACTED_HEATMAPS} \
	--weights  ${BASE_PRETRAINED} \
	--downsampling-factor ${DOWNSAMPLING_FACTOR} \
	--model 'dla_34'