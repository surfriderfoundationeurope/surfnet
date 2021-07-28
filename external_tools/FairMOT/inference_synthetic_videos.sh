export CUDA_VISIBLE_DEVICES=2

dataset_dir='/home/infres/chagneux/datasets/surfrider_data/video_dataset/synthetic_videos/raw/val'
cd ${dataset_dir}
for f in *.MP4; do
echo $f
cd ~/repos/FairMOT
python src/demo.py mot --load_model exp/mot/surfrider_dla34_140_epochs/model_140.pth --conf_thres 0.4 --input-video ${dataset_dir}/$f --output-root ./surfrider_synthetic_videos 
python remap_ids.py --input_file  ./surfrider_synthetic_videos/results.txt --min_len_tracklet 2 --output_name $f
# mv ./surfrider_synthetic_videos/results.txt ./surfrider_synthetic_videos/results.txt 
done 

