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
conda activate surfnet 
```

### Variables
```shell
cd scripts_for_experiments
sh init_shell_variables.sh
```
Then edit the variables according to your machine in 'shell_variables.sh'



