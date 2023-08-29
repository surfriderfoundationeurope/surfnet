import numpy as np
import os
import yaml
from sklearn.model_selection import ParameterGrid
import subprocess
import optuna
import pandas as pd
import mlflow
import mlflow.pytorch

# execute function for several parameters


def get_training_name(params):
    """
    Generate a unique training name based on hyperparameters.

    Args:
        params (dict): Dictionary of hyperparameters.

    Returns:
        str: Unique training name.
    """
    p = [f'{params[i]}{i}' for i in params.keys()]
    return f'yolov5_100imgs_{"_".join(p)}'


def get_training_command(params):
    """
    Construct a training command for the YOLOv5 model using specified hyperparameters.

    Args:
        params (dict): Dictionary of hyperparameters.

    Returns:
        str: Training command.
    """
    return f'python train.py --img 640 --hyp "data/hyps/hyp_surfnet_test.yaml" --batch {params["b"]} --epochs 1  --data "../data_100imgs/data.yaml" --weights "yolov5s.pt" --workers 7 --project "hiba" --name {get_training_name(params)}  --exist-ok'


def get_validation_command(params):
    """
    Construct a validation command for the YOLOv5 model using specified hyperparameters.

    Args:
        params (dict): Dictionary of hyperparameters.

    Returns:
        str: Validation command.
    """
    name = get_training_name(params)
    return f'python val_surfnet.py --img 640 --batch {params["b"]} --data "../data_100imgs/data.yaml" --weights "../yolov5/hiba/{name}/weights/best.pt" --workers 7 --project "validation_res" --name {name}'


def modify_hyp_data(data, params):
    """
    Modify hyperparameters in a YAML file to match the specified parameters.

    Args:
        data (dict): Dictionary of hyperparameters loaded from a YAML file.
        params (dict): Dictionary of hyperparameters to be modified.

    Returns:
        dict: Modified hyperparameters.
    """
    # the parameters that do not belong in the hyp_surfnet.yaml file
    otherParams = ["b", "e"]

    # the parameters that exist in the hyp_surfnet.yaml file
    for p in params:
        if p not in otherParams:
            data[p] = params[p]
    return data


def modify_hyp_file(params):
    """
    Create a modified hyperparameters YAML file.

    Args:
        params (dict): Dictionary of hyperparameters.

    Returns:
        None
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    old_hyp_path = os.path.join(current_dir, 'data/hyps/hyp_surfnet.yaml')
    new_hyp_path = os.path.join(current_dir, 'data/hyps/hyp_surfnet_optuna.yaml')

    # Load the YAML file
    with open(old_hyp_path, 'r') as f:
        data = yaml.safe_load(f)
    # Modify the values
    data = modify_hyp_data(data, params)
    # Save the modified YAML file
    with open(new_hyp_path, 'w') as f:
        yaml.dump(data, f)


def log_mlflow(experiment_name, param_dict, metric_dict, files_dict):
    """
    Log hyperparameters, metrics, and files using MLflow for experiment tracking.

    Args:
        experiment_name (str): Name of the experiment in MLflow.
        param_dict (dict): Dictionary of hyperparameters to log.
        metric_dict (dict): Dictionary of metrics to log.
        files_dict (dict): Dictionary of files to log as artifacts.

    Returns:
        None
    """
    # Initialize MLflow
    #mlflow.set_tracking_uri("/Users/hibatouderti/mlruns")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        # Set your desired parameters
        for param in param_dict:
            mlflow.log_param(param, param_dict[param])

        # Validate the model and log the metric
        for metric in metric_dict:
            mlflow.log_metric(metric, metric_dict[metric])

        # Log the CSV file as an artifact
        for artifact in files_dict:
            mlflow.log_artifact(files_dict[artifact])


def train_yolo(params):
    """
    Train the YOLO model with specified hyperparameters.

    Args:
        params (dict): Dictionary of hyperparameters.

    Returns:
        float: Maximum F1 score.
        float: Corresponding confidence value.
    """
    print("start train_yolo function")
    # modify the hyperparameters yaml file
    experiment_name = get_training_name(params)
    modify_hyp_file(params)

    # run the training command
    cmd = get_training_command(params)
    output = subprocess.check_output(cmd, shell=True)

    print("end training ")

    # run the validation command
    cmd = get_validation_command(params)
    output = subprocess.check_output(cmd, shell=True)

    print("end validation ")

    # find the f1, confidence values
    files_dict = {}
    try:

        df = pd.read_csv(f'validation_res/{experiment_name}/F1_results.csv')
        f1 = df['f1'].tolist()
        confidence = df['confidence'].tolist()
        max_f1, max_conf = f1.max(), confidence[f1.argmax()]

        files_dict["f1_csv"] = f'validation_res/{experiment_name}/F1_results.csv'
        files_dict["f1_png"] = f'validation_res/{experiment_name}/F1_curve.png'

    except:
        max_f1, max_conf = 0, 0
    finally:
        metric_dict = {"f1": max_f1, "confidence": max_conf}
        files_dict["hyp.yaml"] = f'hiba/{experiment_name}/hyp.yaml'
        files_dict["confusion_matrix.png"] = f'validation_res/{experiment_name}/confusion_matrix.png'

        log_mlflow(experiment_name, params, metric_dict, files_dict)
        print("end mlflow logging ")
        print("end train_yolo function")
        return max_f1, max_conf


def multi_objective(trial):
    """
    Objective function for multi-objective optimization using Optuna.
    
    Args:
        trial: Optuna trial object.

    Returns:
        float: Maximum F1 score.
        float: Corresponding confidence value.
    """
    print("start objective function")
    # Define hyperparameter search spaces using Optuna's suggest methods
    initial_learning_rate = trial.suggest_float('lr0', 1e-5, 1e-1, log=True)
    final_learning_rate = trial.suggest_float('lrf', 1e-5, 1e-1, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)
    momentum = trial.suggest_float('momentum', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 10, 50)

    # Round the suggested value to 5 digits after the decimal point
    initial_learning_rate = round(initial_learning_rate, 6)
    final_learning_rate = round(final_learning_rate, 6)
    weight_decay = round(weight_decay, 6)
    momentum = round(momentum, 6)

    params = {
        'b': batch_size,
        'lr0': initial_learning_rate,
        'lrf': final_learning_rate,
        'momentum': momentum,
        'weight_decay': weight_decay
    }
    # Train your YOLO model with the suggested hyperparameters
    f1, conf = train_yolo(params)

    print("end objective function")
    return f1, conf


def importance(study):
    """
    Calculate and print the importance of hyperparameters in the Optuna study.

    Args:
        study: Optuna study object.

    Returns:
        None
    """
    # Get the most important parameters
    importance = optuna.importance.get_param_importances(study)
    params = pd.DataFrame.from_dict(importance, orient='index', columns=['importance'])
    params = params.sort_values(by='importance', ascending=False)

    # Print the most important parameters
    print(params)


def objective(trial):
    """
    Objective function for Optuna hyperparameter optimization.
    
    Args:
        trial: Optuna trial object.

    Returns:
        float: Maximum F1 score.
    """
    f1, conf = multi_objective(trial)
    return f1


def multi_objective_optimisation():
    """
    Perform multi-objective optimization using Optuna.
    
    Returns:
        None
    """
    study = optuna.create_study(directions=["maximize", "maximize"])
    study.optimize(multi_objective, n_trials=3)  # Change the number of trials as needed

    # Retrieve the best trials (a list of FrozenTrial objects)
    best_trials = study.best_trials

    # Access the best hyperparameters and evaluation metrics from each best trial
    for i, trial in enumerate(best_trials, 1):
        best_params = trial.params
        best_values = trial.values
        print(f"Best Hyperparameters for Trial {i}: {best_params}")
        print(f"Best Evaluation Metric for Trial {i}: {best_values}")

    print("Number of finished trials: ", len(study.trials))
    importance(study)


def objective_optimisation():
    """
    Perform single-objective optimization using Optuna.
    
    Returns:
        None
    """
    study = optuna.create_study(directions=["maximize"])
    study.optimize(objective, n_trials=1)  # Change the number of trials as needed

    # Use this if you have one parameter to optimise
    best_params = study.best_params
    best_value = study.best_value

    print("Best Hyperparameters:", best_params)
    print("Best F1 value:", best_value)
    print("Number of finished trials: ", len(study.trials))
    #importance(study)


def main():
    """
    Main function to initiate hyperparameter optimization.
    
    Returns:
        None
    """
    objective_optimisation()


if __name__ == "__main__":
    main()
