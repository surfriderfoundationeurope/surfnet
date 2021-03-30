# Surfnet


## Installation 
### Conda environments

```shell
conda env create --name surfnet --file environment.yml
conda activate surfnet 
```

### Data and paths 
If you don't have data already, you can download some using the following scripts:

```shell
cd data
sh download_surfrider_images.sh
sh download_synthetic_videos.sh
sh download_synthetic_objects.sh
```

Then run the following scripts to initialize the correct paths: 

```shell
cd scripts_for_experiments
sh init_shell_variables.sh
```

Alternatively, if you have data stored somewhere else, modify the newly created shell_variables.sh script to point to the location of your convenience, or create symlinks to the folders of the repo. 

### Building DCN dependencies (if using CenterNet)

You need to build CUDA compiled operators for the deformable convolutions module used in CenterNet models: 
```shell 
cd base/centernet/networks
rm -rf DCNv2
git clone https://github.com/jinfagang/DCNv2_latest.git
mv DCNv2_latest/ DCNv2/ 
cd DCNv2
./make.sh
```


## Base network

### Training 

You can set a few hyperparameters in scripts_for_experiments/train_base.sh and run it 

```shell
sh scripts_for_experiments/train_base.sh
```

Then open Tensorboard in your browser to follow training. 

For more control refer to the parse_args() function in train_base.py. 

## Extension network 

### Obtaining heatmaps from the base network 
If you need a pretrained version of the base network you can run: 

```shell 
cd external_pretrained_models
sh download_pretrained_base.sh 
```
This will download one of our pretrained models. 

To extract heatmaps from the base network: 

```shell 
sh scripts_for_experiments/extract_heatmaps.sh
``` 

You can modify the BASE_PRETRAINED variable in scripts_for_experiments/shell_variables.sh to point to the base networks weights that you obtained yourself through retraining, for example.
 
 
 ### Training 
 
Similarly as for the base network, you can run:

```shell
sh scripts_for_experiments/train_extension.sh
```

Then open Tensorboard in your browser to follow training.


## Results & Results

A few plot / evaluation functions are available in tests.py. 
