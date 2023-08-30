import yaml
from sklearn.model_selection import ParameterGrid
import subprocess

# execute function for several parameters


def get_training_name(params):
    """
    Generate a unique training name based on the provided parameters.

    Args:
        params (dict): Dictionary containing hyperparameters.

    Returns:
        str: A unique training name.
    """
    p = [f'{params[i]}{i}' for i in params.keys()]
    return f'yolov5_100imgs_{"_".join(p)}'


def get_command(params):
    """
    Generate a training command based on the provided hyperparameters.

    Args:
        params (dict): Dictionary containing hyperparameters.

    Returns:
        str: Training command.
    """
    return f'python train.py --img 640 --hyp "data/hyps/hyp_surfnet_test.yaml" --batch {params["b"]} --epochs {params["e"]}   --data "../data_100imgs/data.yaml" --weights "yolov5s.pt" --workers 7 --project "hiba" --name {get_training_name(params)}  --exist-ok'


def modify_hyp_data(data, params):
    """
    Modify hyperparameter data in a YAML file based on provided parameters.

    Args:
        data (dict): Hyperparameter data from a YAML file.
        params (dict): Dictionary containing hyperparameters to be modified.

    Returns:
        dict: Modified hyperparameter data.
    """
    otherParams = ["b", "e"]
    for p in params:
        if p not in otherParams:
            data[p] = params[p]
    return data

# add hyp to params


def modify_hyp_file(params):
    """
    Add hyperparameters to a YAML file based on provided parameters.

    Args:
        params (dict): Dictionary containing hyperparameters to be added.

    Returns:
        None
    """
    # Load the YAML file
    with open('data/hyps/hyp_surfnet.yaml', 'r') as f:
        data = yaml.safe_load(f)
    # Modify the values
    data = modify_hyp_data(data, params)
    # Save the modified YAML file
    with open('data/hyps/hyp_surfnet_test.yaml', 'w') as f:
        yaml.dump(data, f)


# Define the hyperparameters to tune
param_grid = {
    'e': [50],  # epochs
    'b': [40],  # batch_size
    'lr0': [0.01, 0.05, 0.1],
    'lrf': [0.01, 0.05, 0.1],
    # 'momentum': [0.9, 0.937  , 0.95 ],
    # 'weight_decay': [0.0001, 0.0005, 0.001]
}

# Generate all possible combinations of hyperparameters
param_combinations = ParameterGrid(param_grid)

# Loop through each combination and execute a shell command
for params in param_combinations:
    print(params)
    modify_hyp_file(params)
    cmd = get_command(params)
    # Run the shell command and capture the output
    output = subprocess.check_output(cmd, shell=True)
    # Print the best hyperparameters found by the grid search
    # print(f"For hyperparameters: {params}, best hyperparameters: {output}")
