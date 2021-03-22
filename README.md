# Surfnet


## Installation 
### Conda environments

#### For use with Deeplab models 
```shell
conda env create --name surfnet --file environment.yml
conda activate surfnet 
```

#### For use with CenterNet models 
```shell
conda env create --name surfnet_DLA --file environment_centernet_DCN.yml
conda activate surfnet_DLA 
```

### Variables
```shell
cd scripts_for_experiments
sh init_shell_variables.sh
```
Then edit the variables according to your machine in 'shell_variables.sh'


### Building DCN dependencies (if using CenterNet)

```shell 
cd base/centernet/networks
rm -rf DCNv2
git clone https://github.com/CharlesShang/DCNv2.git
cd DCNv2
git fetch origin pull/92/head:pytorch17
git checkout pytorch17
./make.sh
```
