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

def parse_args():
    parser = argparse.ArgumentParser(description='Enzyme Binary Classification')

    # NAMING CONVENTION
    parser.add_argument('--run_name', type=str, default='isEnzyme', help='Name for the run, used in logging')
    parser.add_argument('--run_no', type=int, default=1000, help='param group')
    parser.add_argument('--pooling_method', type=str, required=True, help='Directory of protein embeddings')
    
    # HYPERPARAMS
    parser.add_argument('--hyperparam_group', type=int, default=-1, help='hyperparam group')
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


    # OTHER
    parser.add_argument("--other_notes", type=str, default=None, help='Notes about this run')

        
    
    # PATHS DEFAULT
    parser.add_argument('--train_path', type=str, 
                        default = "data_enzyme_or_not/train_set_enzyme_or_not.csv", 
                        help='Path to the training data file')
    parser.add_argument('--val_path', type=str, 
                        default = "data_enzyme_or_not/val_set_enzyme_or_not.csv", 
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
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training')
    
    # DEBUG and TEST
    parser.add_argument('--rapid_debug_mode', action='store_true', help='Run in debug mode with minimal data')
    parser.add_argument('--eval_only', action='store_true', help='Run only evaluation on the test set')
    
    
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
            args.exponent_class_weight = group_params['exp_class_weight']
        else:
            print(f"No hyperparameter group found for {args.hyperparam_group}. Using defaults.", flush=True)

    # Print all argparse variables
    for arg, value in vars(args).items():
        print(f'{arg}: {value}', flush=True)

    return args

def setup_logging(args):
    log_dir = f"logs/{args.pooling_method}"
    os.makedirs(log_dir, exist_ok=True)
    if args.other_notes is None:
        file_name = f"{args.run_no}_log.txt"
    else:
        file_name = f"{args.run_no}_{args.other_notes}_log.txt"
    file_path = os.path.join(log_dir, file_name)
    logging.basicConfig(filename=file_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging
