export CUDA_VISIBLE_DEVICES=1
cd src
nohup python train.py mot \
    --exp_id surfrider_1070_images_290_epochs \
    --gpus 0 \
    --batch_size 16 \
    --load_model '../models/ctdet_coco_dla_2x.pth' \
    --num_epochs 290 \
    --lr_step '140' \
    --data_cfg '../src/lib/cfg/surfrider.json' \
    --data_dir '../src/data/surfrider' &