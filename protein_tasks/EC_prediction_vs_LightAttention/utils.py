import torch
import numpy as np
import random
import argparse
import os
import logging  # Import the logging module
import pandas as pd

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def inject_hyperparameters(args, hyperparams):
    """
    Dynamically inject hyperparameters into argparse arguments.
    """
    args.max_LR = hyperparams.get("max_LR",
                                  args.max_LR)
    args.weight_decay = hyperparams.get("weight_decay",
                                        args.weight_decay)
    args.dropout_rate = hyperparams.get("dropout_rate",
                                   args.dropout_rate)
    args.reduce_lr_ratio = hyperparams.get("reduce_lr_ratio",
                                           args.reduce_lr_ratio)
    

    
    return args

def parse_args(verbose=False):
    parser = argparse.ArgumentParser(description='Enzyme Binary Classification')

    # NAMING CONVENTION
    parser.add_argument('--run_name', type=str, default='isEnzyme', help='Name for the run, used in logging')
    parser.add_argument('--run_no', type=int, default=1000, help='param group')
    parser.add_argument('--pooling_method', type=str, required=False, help='Directory of protein embeddings')
    parser.add_argument('--PLM', type=str, required=True, help='Directory of protein embeddings')
    parser.add_argument('--data_fraction', type=float, default=1, help='Fraction of data to use')
    
    
    # HYPERPARAMS
    parser.add_argument('--hyperparam_group', type=int, default=-1, help='hyperparam group')
    parser.add_argument('--optuna_hparam_trial', type=int, default=-1, help='which trial to pick hyperparams from')
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Initial learning rate')
    parser.add_argument('--max_LR', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (L2 penalty)')
    parser.add_argument('--dropout_rate', type=float, default=0.05, help='Dropout rate')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--clip_value', type=float, default=1.0, help='Max norm of the gradients')
    parser.add_argument('--exponent_class_weight', type=float, default=1.0, help='Power to adjust the class weights')
    # for early stop
    parser.add_argument('--patience_es', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--patience_lr', type=int, default=4, help='Patience for early stopping')
    parser.add_argument('--reduce_lr_ratio', type=float, default=0.5, help='Reduce LR by this much if early stopping and reduceLRPlateau')
    parser.add_argument("--slopeLeakyReLU", type=float, default=0.1, help="slope of leaky relu")
    parser.add_argument('--percent_epochs_lr_rise', type=float, default=0.1, help='The percentage of epochs where LR rises for oneCycleScheduler')
    parser.add_argument("--bayesopt_step_num", type=int, default=0, help="Steps to take for bayesian hyperparam opt")


    # OTHER
    parser.add_argument("--other_notes", type=str, default=None, help='Notes about this run')
    parser.add_argument('--val_fold_index', type=int, default=1, help='validation fold index')
    parser.add_argument('--total_fold_no', type=int, default=5, help='Patience for early stopping')


        
    
    # PATHS DEFAULT
    parser.add_argument('--train_path', type=str, 
                        default = "data_enzyme_or_not/train_set_enzyme_or_not.csv", 
                        help='Path to the training data file')
    parser.add_argument('--val_path', type=str, 
                        default = "data_enzyme_or_not/val_set_enzyme_or_not.csv", 
                        help='Path to the test data file')
    parser.add_argument('--trainval_path', type=str, 
                        default = "data_enzyme_or_not/trainval_set.csv", 
                        help='Path to the test data file')
    parser.add_argument('--test_path_human', type=str, 
                        default = "data_enzyme_or_not/test_set_enzyme_or_not_human.csv",
                        help='Path to the test data file')
    parser.add_argument('--test_path_mammalian', type=str, 
                        default = "data_enzyme_or_not/test_set_enzyme_or_not_mammalian.csv",
                        help='Path to the test data file')

    parser.add_argument('--test_path_vertebrate', type=str, 
                        default = "data_enzyme_or_not/test_set_enzyme_or_not_vertebrate.csv",
                        help='Path to the test data file')

    parser.add_argument('--test_path_animal', type=str, 
                        default = "data_enzyme_or_not/test_set_enzyme_or_not_animal.csv",
                        help='Path to the test data file')

    parser.add_argument('--test_path_eukaryote', type=str, 
                        default = "data_enzyme_or_not/test_set_enzyme_or_not_eukaryote.csv",
                        help='Path to the test data file')
   
    
    
    
    # BOILERPLATE
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='Input batch size for training')
    
    # DEBUG and TEST
    parser.add_argument('--rapid_debug_mode', action='store_true', help='Run in debug mode with minimal data')
    parser.add_argument('--eval_only', action='store_true', help='Run only evaluation on the test set')
    parser.add_argument('--use_wandb', action='store_true', help='Monitor job on wandb')
    
    
    args = parser.parse_args()

    if args.hyperparam_group > -0.5:
        # Read hyperparameters from CSV
        df_params = pd.read_csv('hyperparams.csv', index_col='hyperparam_group')
        
        if args.hyperparam_group in df_params.index:
            # Update the arguments with the parameters from the CSV
            group_params = df_params.loc[args.hyperparam_group]
            args.max_LR = group_params['max_lr']
            args.weight_decay = group_params['weight_decay']
            args.dropout_rate = group_params['dropout']
            args.reduce_lr_ratio = float(group_params['reduce_lr_ratio'])
        else:
            print(f"No hyperparameter group found for {args.hyperparam_group}. Using defaults.", flush=True)

    elif args.optuna_hparam_trial > -1:
            df_params = pd.read_csv(f'./hpt_opt/optuna_hyperparams.csv')
            if not args.optuna_hparam_trial in list(df_params['trial_number']):
                raise Exception(f"You've provided a wrong optuna_hparam_trial value. You provided {args.optuna_hparam_trial} but the ones we have are {df_params['trial_number'].unique()}")

            group_params = df_params[df_params['trial_number'] == args.optuna_hparam_trial]
            args.max_LR = float(group_params['max_LR'])
            args.weight_decay = float(group_params['weight_decay'])
            args.dropout_rate = float(group_params['dropout_rate'])
            args.reduce_lr_ratio = float(group_params['reduce_lr_ratio'])

    if verbose:
        # Print all argparse variables
        for arg, value in vars(args).items():
            print(f'{arg}: {value}', flush=True)

    return args

def setup_logging(args, hparam=-1):
    log_dir = f"logs/{args.pooling_method}"
    os.makedirs(log_dir, exist_ok=True)
    if args.other_notes is None:
        file_name = f"{args.run_no}_{args.run_name}_hparam{hparam}_{args.data_fraction}_log.txt"
    else:
        file_name = f"{args.run_no}_{args.run_name}_{args.other_notes}_hparam{hparam}_{args.data_fraction}_log.txt"
    file_path = os.path.join(log_dir, file_name)
    logging.basicConfig(filename=file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging


def save_model(model_LA, model_predictor, path):
    checkpoint = {
        'model_LA_state_dict': model_LA.state_dict(),
        'model_predictor_state_dict': model_predictor.state_dict()
    }
    torch.save(checkpoint, path)

        
def load_model(model_LA, model_predictor, path, device):
    checkpoint = torch.load(path, map_location=device)
    model_LA.load_state_dict(checkpoint['model_LA_state_dict'])
    model_predictor.load_state_dict(checkpoint['model_predictor_state_dict'])
    return model_LA, model_predictor
