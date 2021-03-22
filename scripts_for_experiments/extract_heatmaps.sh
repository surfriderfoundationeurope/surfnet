. scripts_for_experiments/shell_variables.sh

python extract_and_save_heatmaps.py \
	--video-dir ${SYNTHETIC_VIDEOS_PATH} \
	--output-dir ${BASE_EXTRACTED_HEATMAPS} \
	--weights  ${BASE_PRETRAINED} \
	--downsampling-factor ${DOWNSAMPLING_FACTOR}