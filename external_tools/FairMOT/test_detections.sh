export CUDA_VISIBLE_DEVICEs=0
cd src 
python test_det.py mot \
    --load_model ../exp/mot/surfrider_1070_images_290_epochs/model_290.pth \
    --data_cfg '../src/lib/cfg/surfrider.json' \
    --data_dir '../src/data/surfrider' \
    --not_reg_offset