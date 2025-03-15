import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from data import get_loader_LA, get_loader_with_split_LA
from models import LightAttention, ECPredictor
from utils import set_seed, parse_args, setup_logging, inject_hyperparameters, save_model, load_model
import utils
import wandb
import csv
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import os
from tqdm import tqdm
import optuna
from datetime import datetime
import time
import random
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score

def train(model_LA, model_predictor, train_loader, optimizers, schedulers, device, class_weights, args):
    model_LA.train()
    model_predictor.train()
    total_loss = 0

    if args.rapid_debug_mode:
        batch_counter = 0
        max_batch = 50

    # Ensure class weights are on the correct device
    class_weights = class_weights.to(device)  # Move class weights to the correct device
    
    for data, mask, target in train_loader:  # Expect three values
    
        if args.rapid_debug_mode:
            batch_counter += 1
            if batch_counter > max_batch:
                break

            print(f"In train batch {batch_counter}\ndata has shape {data.shape} and is\n{data}\n")
        
        data, mask, target = data.to(device), mask.to(device), target.to(device)
        optimizers[0].zero_grad()
        optimizers[1].zero_grad()

        rep = model_LA(data, mask)#.to_device()
        output = model_predictor(rep)#.to(device)
        loss = F.cross_entropy(output, target.long(), weight=class_weights.to(device))

        loss.backward()

        
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model_LA.parameters(), args.clip_value)
        torch.nn.utils.clip_grad_norm_(model_predictor.parameters(), args.clip_value)
        
        optimizers[0].step()
        optimizers[1].step()
        
        if not schedulers is None:
            schedulers[0].step()
            schedulers[1].step()
        total_loss += loss.item()
    return total_loss / len(train_loader)



def test(model_LA, model_predictor, test_loader, device, args):
    
    model_LA.eval()
    model_predictor.eval()
    total_preds, total_targets = [], []
    predictions = []

    with torch.no_grad():
        for data, mask, targets in test_loader:
            data, mask, targets = data.to(device), mask.to(device), targets.to(device)
            #outputs = model(data)
            rep = model_LA(data, mask)#.to_device()
            outputs = model_predictor(rep)#.to(device)
            probs = torch.softmax(outputs, dim=1).cpu()  # Convert logits to probabilities
            predicted_labels = torch.argmax(probs, dim=1)  # Pick the class with highest probability

            predictions.extend(zip(probs.tolist(), predicted_labels.tolist(), targets.cpu().tolist()))
            total_preds.extend(predicted_labels.tolist())  # Store predicted class indices
            total_targets.extend(targets.cpu().tolist())  # Store actual class labels

    # Calculate metrics
    accuracy = accuracy_score(total_targets, total_preds)
    balanced_acc = balanced_accuracy_score(total_targets, total_preds)
    cohen_kappa = cohen_kappa_score(total_targets, total_preds)

    return predictions, accuracy, balanced_acc, cohen_kappa


def validate(model_LA, model_predictor, val_loader, device, class_weights, args):
    model_LA.eval()
    model_predictor.eval()
    total_loss = 0
    total_preds, total_targets = [], []

    with torch.no_grad():
        batch_index = 0
        for data, mask, target in val_loader:
            data, mask, target = data.to(device), mask.to(device), target.to(device)
            #output = model(data)
            rep = model_LA(data, mask)#.to_device()
            output = model_predictor(rep)#.to(device)

            # Compute loss using cross-entropy for multi-class classification
            try:
                loss = F.cross_entropy(output, target)
            except Exception as e:
                print(f"Error in batch index {batch_index}")
                print(f"data.shape: {data.shape}")
                print(f"output.shape: {output.shape}, target.shape: {target.shape}")
                raise Exception(e)

            total_loss += loss.item()

            # Collect predictions and targets
            probs = torch.softmax(output, dim=1).cpu()  # Convert logits to probabilities
            predicted_labels = torch.argmax(probs, dim=1)  # Pick the class with highest probability
            
            total_preds.extend(predicted_labels.tolist())
            total_targets.extend(target.cpu().tolist())
            batch_index += 1

    # Calculate metrics
    accuracy = accuracy_score(total_targets, total_preds)
    balanced_acc = balanced_accuracy_score(total_targets, total_preds)
    cohen_kappa = cohen_kappa_score(total_targets, total_preds)

    # Return metrics along with average loss
    return total_loss / len(val_loader), accuracy, balanced_acc, cohen_kappa



def main():
    args = parse_args()
    set_seed(args.seed)
    logging = setup_logging(args)

    
    cv_fold = random.randint(1, 7)
    train_loader, val_loader = get_loader_with_split_LA(
            file_path = args.trainval_path, 
            h5_path = f'./h5_files/trainval_{args.PLM}.h5', 
            val_fold_index = cv_fold,
            args=args,
            data_fraction = args.data_fraction
        )



    test_loader_human = get_loader_LA(file_path=args.test_path_human, 
                                 h5_path = f'./h5_files/test_{args.PLM}.h5', 
                                 args = args,
                                 shuffle=True)


    test_loader_mammalian = get_loader_LA(file_path=args.test_path_mammalian, 
                                 h5_path = f'./h5_files/test_{args.PLM}.h5', 
                                 args=args,
                                 shuffle=True)

    test_loader_vertebrate = get_loader_LA(file_path=args.test_path_vertebrate, 
                                 h5_path = f'./h5_files/test_{args.PLM}.h5', 
                                 args=args,
                                 shuffle=True)

    test_loader_animal = get_loader_LA(file_path=args.test_path_animal, 
                                 h5_path = f'./h5_files/test_{args.PLM}.h5', 
                                       args=args,
                                 shuffle=True)

    test_loader_eukaryote = get_loader_LA(file_path=args.test_path_eukaryote, 
                                 h5_path = f'./h5_files/test_{args.PLM}.h5', 
                                 args=args, 
                                 shuffle=True)

    

    
    class_weights = train_loader.dataset.class_weights

    if args.rapid_debug_mode:
        print(f"class_weights {class_weights}", flush=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}", flush=True)
    if args.PLM == 'protbert':
            input_dim = 1024
    elif args.PLM == 'esm2':
        input_dim = 1280


    model_LA = LightAttention(embeddings_dim = input_dim,
                             dropout = args.dropout_rate,
                             conv_dropout = args.dropout_rate).to(device)
    
    model_predictor = ECPredictor(input_dim*2, args.hidden_dim, 6, dropout_rate=args.dropout_rate, negative_slope=args.slopeLeakyReLU).to(device)

   

    optimizer_LA = optim.AdamW(model_LA.parameters(), lr=args.max_LR, weight_decay=args.weight_decay)
    optimizer_predictor = optim.AdamW(model_predictor.parameters(), lr=args.max_LR, weight_decay=args.weight_decay)

    optimizers = [optimizer_LA, optimizer_predictor]
    
    scheduler_LA = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_LA, mode='max', 
                                                           factor=args.reduce_lr_ratio, 
                                                           patience=args.patience_lr, 
                                                           verbose=True)
    scheduler_predictor = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_predictor, mode='max', 
                                                           factor=args.reduce_lr_ratio, 
                                                           patience=args.patience_lr, 
                                                           verbose=True)

    schedulers = [scheduler_LA, scheduler_predictor]
    
    
    # Replace OneCycleLR with a scheduler more compatible with early stopping
    
    print("initialized ReduceLROnPlateau scheduler", flush=True)
   

    best_val_loss = float('inf')
    best_val_metric = -5
    epochs_since_improvement = 0  # Track the number of epochs since last improvement for early stopping
    os.makedirs(f"best_models_LA/{args.pooling_method}", exist_ok=True)

    
    best_model_path = os.path.join(f"best_models_LA", f'model-run{args.run_no}_{args.run_name}_{args.data_fraction}PercentData.pt')


    epochs = args.num_epochs
    
    if not args.eval_only:
        if not args.rapid_debug_mode:
            pass
        if args.use_wandb:
            wandb.init(project="poolingEnzyme", name= f"{args.pooling_method}-run{args.run_no}", config=vars(args))
        
        for epoch in range(epochs):
            if args.early_stopping: # matter of whether you pass the scheduler or not
                #def train(model_LA, model_predictor, train_loader, optimizers, schedulers, device, class_weights, args)
                train_loss = train(model_LA, model_predictor, train_loader, optimizers=optimizers, schedulers=None, device=device, class_weights=class_weights, args=args)
            else:
                train_loss = train(model_LA, model_predictor, train_loader, optimizers=optimizers, schedulers=schedulers, device=device, class_weights=class_weights, args=args)
            #val_loss = validate(model, val_loader, device, class_weights, args)
            #val_loss, val_auroc, val_auprc, val_f1, val_mcc = validate(model, val_loader, device, class_weights, args)
            val_loss, val_acc, val_bal_acc, val_qwk = validate(model_LA, model_predictor, val_loader, device, class_weights, args)
            val_metric = val_acc + val_bal_acc + val_qwk
            if args.early_stopping:
                schedulers[0].step(val_metric)
                schedulers[1].step(val_metric)


            current_lr = optimizers[0].param_groups[0]['lr']
            
            if not args.rapid_debug_mode:
                logging.info(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
            if not args.rapid_debug_mode:
                pass
            if args.use_wandb:
                wandb.log({"train_loss": train_loss, 
                           "val_loss": val_loss,
                           "current_lr": current_lr,
                          "epoch": epoch})

            # Save the model if it has the best validation loss
            if val_metric > best_val_metric:
                best_val_metric = val_metric
                #torch.save(model.state_dict(), best_model_path)
                save_model(model_LA, model_predictor, best_model_path)
                print(f"Saved new best model with validation loss: {val_loss} at epoch {epoch}", flush=True)
                epochs_since_improvement = 0  # reset counter
            else:
                epochs_since_improvement += 1
                #print(f"No improvement in validation loss for {epochs_since_improvement} epochs")

            # Early stopping condition
            if args.early_stopping and epochs_since_improvement >= args.patience_es:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Load the best model for testing
    #model.load_state_dict(torch.load(best_model_path))
    model_LA, model_predictor = load_model(model_LA, model_predictor, best_model_path, device)

    tests = ["human", "mammalian", "vertebrate", "animal", "eukaryote"]
    test_loaders = [test_loader_human, test_loader_mammalian, test_loader_vertebrate, test_loader_animal, test_loader_eukaryote]

    for i in range(len(tests)):
        group = tests[i]
        test_loader = test_loaders[i]
    

        # Testing and logging results
        #test_predictions, auroc, auprc, f1, mcc = test(model, test_loader, device, args)
        predictions, accuracy, balanced_acc, cohen_kappa = test(model_LA, model_predictor, test_loader, device, args)
        if args.other_notes is None:
            prediction_file = f"run{args.run_no}_{args.run_name}_{args.data_fraction*100}percentData_pred_{group}.csv"
        else:
            prediction_file = f"run{args.run_no}_{args.run_name}_{args.data_fraction*100}percentData_{args.other_notes}_pred_{group}.csv"

        dir_pred = f"predictions_LA"
        os.makedirs(dir_pred, exist_ok=True)
        
        with open(f'{dir_pred}/{prediction_file}', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Probability', 'Predicted Label', 'Actual Label'])
            writer.writerows(predictions)
        
        # Log metrics to console and wandb
        metrics = {
            f"test_accuracy_{group}": accuracy,
            f"test_balanced_acc_{group}": balanced_acc,
            f"test_cohen_kappa_{group}": cohen_kappa
        }
    
        logging.info(f'Test Metrics_{group}: {metrics}\n\n')
        if args.use_wandb:
            wandb.log(metrics)


def objective(trial):
    # Start timing
    start_time = time.time()
    args = parse_args(verbose=True)
    args = parse_args()
    set_seed(args.seed)
    logging = setup_logging(args)

    # Define search space
    max_LR = trial.suggest_float("max_LR", 0.0009, 0.011, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.009, 0.011)
    dropout_rate = trial.suggest_float("dropout_rate", 0.04, 0.26)
    reduce_lr_ratio = trial.suggest_float("reduce_lr_ratio", 0.19, 0.51)
    exponent_class_weight = trial.suggest_float("exponent_class_weight", 0.74, 2.01)

    # Parse arguments and inject hyperparameters
    args = inject_hyperparameters(args, {
        "max_LR": max_LR,
        "weight_decay": weight_decay,
        "dropout_rate": dropout_rate,
        "reduce_lr_ratio": reduce_lr_ratio,
        "exponent_class_weight": exponent_class_weight
    })

    print(f"\n\n\nThe current trial hyperparams are:")
    print(f"max_LR {args.max_LR}")
    print(f"weight_decay {args.weight_decay}")
    print(f"dropout_rate {args.dropout_rate}")
    print(f"reduce_lr_ratio {args.reduce_lr_ratio}")
    print(f"exponent_class_weight {args.exponent_class_weight}\n\n")

    list_best_val_loss = []
    list_best_val_auroc = []
    list_best_val_auprc = []
    list_best_val_f1 = []
    list_best_val_mcc = []
    list_best_val_metric = []
    list_epochs_ran = []
    failed_fold_no = 0


    for cv_fold in tqdm(range(1, args.total_fold_no+1)):

        best_val_loss = float('inf')
        best_val_auroc = 0
        best_val_auprc = 0
        best_val_f1 = 0
        best_val_mcc = -1
        best_val_metric = -5


        train_loader, val_loader = get_loader_with_split_LA(
            file_path = args.trainval_path, 
            h5_path = f'./h5_files/trainval_{args.PLM}.h5', 
            val_fold_index = cv_fold,
            args=args,
            data_fraction = data_fraction
        )

        class_weights = train_loader.dataset.class_weights
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device is {device}", flush=True)
        if args.PLM == 'protbert':
            input_dim = 1024
        elif args.PLM == 'esm2':
            input_dim = 1280

        model_LA = LightAttention(embeddings_dim = input_dim,
                             dropout = args.dropout_rate,
                             conv_dropout = args.dropout_rate).to(device)
        
        model_predictor = ECPredictor(input_dim*2, args.hidden_dim, 6, dropout_rate=args.dropout_rate, negative_slope=args.slopeLeakyReLU).to(device)

    
        
        # Replace OneCycleLR with a scheduler more compatible with early stopping
        optimizer_LA = optim.AdamW(model_LA.parameters(), lr=args.max_LR, weight_decay=args.weight_decay)
        optimizer_predictor = optim.AdamW(model_predictor.parameters(), lr=args.max_LR, weight_decay=args.weight_decay)
    
        optimizers = [optimizer_LA, optimizer_predictor]
        
        scheduler_LA = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_LA, mode='max', 
                                                               factor=args.reduce_lr_ratio, 
                                                               patience=args.patience_lr, 
                                                               verbose=True)
        scheduler_predictor = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_predictor, mode='max', 
                                                               factor=args.reduce_lr_ratio, 
                                                               patience=args.patience_lr, 
                                                               verbose=True)

        schedulers = [scheduler_LA, scheduler_predictor]
        
        print("initialized ReduceLROnPlateau scheduler", flush=True)
    
    
        epochs_since_improvement = 0  # Track the number of epochs since last improvement for early stopping
        os.makedirs(f"best_models/{args.pooling_method}", exist_ok=True)
    
        
        best_model_path = os.path.join(f"best_models_LA", f'model-run{args.run_no}.pt')
    
        if args.other_notes is None:
            best_model_path = os.path.join(f"best_models_LA", f'model-run{args.run_no}.pt')
        else:
            best_model_path = os.path.join(f"best_models_LA", f'model-run{args.run_no}_{args.other_notes}.pt')
            #file_name = f"{args.run_no}_{args.other_notes}_log.txt"
    
        epochs = args.num_epochs
        if not args.eval_only:
            if not args.rapid_debug_mode:
                pass
            if args.use_wandb:
                wandb.init(project="poolingEnzyme", name= f"{args.pooling_method}-run{args.run_no}", config=vars(args))
            
            for epoch in range(epochs):
                if args.early_stopping: # matter of whether you pass the scheduler or not
                    train_loss = train(model, train_loader, optimizer=optimizer, schedulers=None, device=device, class_weights=class_weights, args=args)
                else:
                    train_loss = train(model, train_loader, optimizer=optimizer, schedulers=schedulers, device=device, class_weights=class_weights, args=args)
                #val_loss = validate(model, val_loader, device, class_weights, args)
                #val_loss, val_auroc, val_auprc, val_f1, val_mcc = validate(model, val_loader, device, class_weights, args)
                val_loss, val_acc, val_bal_acc, val_qwk = validate(model_LA, 
                                                                   model_predictor, 
                                                                   val_loader, 
                                                                   device, class_weights, args)
                
                val_metric = val_acc + val_bal_acc + val_qwk


    
                if args.early_stopping:
                    schedulers[0].step(val_metric)
                    schedulers[1].step(val_metric)
    
    
                current_lr = optimizers[0].param_groups[0]['lr']
                
                if not args.rapid_debug_mode:
                    logging.info(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
                if not args.rapid_debug_mode:
                    pass
                if args.use_wandb:
                    wandb.log({"train_loss": train_loss, 
                               "val_loss": val_loss,
                               "current_lr": current_lr,
                              "epoch": epoch})
    
                # Save the model if it has the best validation loss
                if val_metric > best_val_metric:
                    best_val_loss = val_loss
                    best_val_auroc = val_auroc
                    best_val_auprc = val_auprc
                    best_val_f1 = val_f1
                    best_val_mcc = val_mcc
                    best_val_metric = val_metric
                    
                    #torch.save(model.state_dict(), best_model_path)
                    save_model(model_LA, model_predictor, best_model_path)
                    
                    print(f"Saved new best model with validation loss: {val_loss} at epoch {epoch}", flush=True)
                    epochs_since_improvement = 0  # reset counter
                else:
                    epochs_since_improvement += 1
                    #print(f"No improvement in validation loss for {epochs_since_improvement} epochs")
    
                # Early stopping condition
                if args.early_stopping and epochs_since_improvement >= args.patience_es:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        list_best_val_loss.append(best_val_loss.cpu().item() if isinstance(best_val_loss, torch.Tensor) else best_val_loss)
        list_best_val_auroc.append(best_val_auroc.cpu().item() if isinstance(best_val_auroc, torch.Tensor) else best_val_auroc)
        list_best_val_auprc.append(best_val_auprc.cpu().item() if isinstance(best_val_auprc, torch.Tensor) else best_val_auprc)
        list_best_val_f1.append(best_val_f1.cpu().item() if isinstance(best_val_f1, torch.Tensor) else best_val_f1)
        list_best_val_mcc.append(best_val_mcc.cpu().item() if isinstance(best_val_mcc, torch.Tensor) else best_val_mcc)
        list_best_val_metric.append(best_val_metric.cpu().item() if isinstance(best_val_metric, torch.Tensor) else best_val_metric)
        list_epochs_ran.append(epoch)
        if list_best_val_auroc[-1] == 0:
            print(f"In trial {trial.number}, fold {cv_fold} failed")
            failed_fold_no += 1


        print(f"In trial {trial.number} cv {cv_fold}", flush=True)
        print(f"epochs_ran {list_epochs_ran[-1]}", flush=True)
        print(f"best_val_loss {list_best_val_loss[-1]}", flush=True)
        print(f"best_val_auroc {list_best_val_auroc[-1]}", flush=True)
        print(f"best_val_auprc {list_best_val_auprc[-1]}", flush=True)
        print(f"best_val_f1 {list_best_val_f1[-1]}", flush=True)
        print(f"best_val_mcc {list_best_val_mcc[-1]}", flush=True)
        print(f"best_val_metric {list_best_val_metric[-1]}", flush=True)
        
    

    elapsed_time_minutes = (time.time() - start_time) / 60
    elapsed_time_per_fold = elapsed_time_minutes/args.total_fold_no

    best_val_auroc_avg = np.mean(list_best_val_auroc)
    best_val_auprc_avg = np.mean(list_best_val_auprc)
    best_val_f1_avg = np.mean(list_best_val_f1)
    best_val_mcc_avg = np.mean(list_best_val_mcc)
    mean_epochs_ran = np.mean(list_epochs_ran)
    
    
    if args.use_wandb:
        wandb.log({
            "trial": trial.number,
            "max_LR": trial.params["max_LR"],
            "weight_decay": trial.params["weight_decay"],
            "dropout_rate": trial.params["dropout_rate"],
            "reduce_lr_ratio": trial.params["reduce_lr_ratio"],
            "exponent_class_weight": trial.params["exponent_class_weight"],
            "best_val_loss_avg": np.mean(list_best_val_loss),
            "best_val_auroc_avg":best_val_auroc_avg,
            "best_val_auprc_avg": best_val_auprc_avg,
            "best_val_f1_avg": best_val_f1_avg,
            "best_val_mcc_avg": best_val_mcc_avg,
            'mean_epochs_ran': mean_epochs_ran,
            'elapsed_time_total_minutes': elapsed_time_minutes,
            'elapsed_time_minutes_per_fold': elapsed_time_per_fold,
            'failed_fold_no': failed_fold_no
        })
    
           
    return best_val_auroc_avg, best_val_auprc_avg, best_val_f1_avg, best_val_mcc_avg
        

            

if __name__ == "__main__":

    args = parse_args()
    time_very_first = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.use_wandb:
        if args.bayesopt_step_num > 0:
            wandb.init(project="optuna_hyperparam_tuning_EC", name=f"{args.run_name}_run{args.run_no}_poolby{args.pooling_method}_{args.PLM}_time{time_very_first}")
        else:
            wandb.init(project="EC", name=f"{args.run_name}_run{args.run_no}_poolby{args.pooling_method}_{args.PLM}_time{time_very_first}")
            pass
    
    
    if args.bayesopt_step_num == 0:
        main()
    else:
        BASE_STORAGE = "./hpt_opt"
        storage = f"sqlite:///{BASE_STORAGE}/optuna_model_study_poolby{args.pooling_method}_{args.PLM}.db"  # Creates an SQLite file named optuna_study.db
        study = optuna.create_study(
            directions= ["maximize", "maximize", "maximize", "maximize"],
            study_name="categorical_optimization_split",  # Name of the study
            storage=storage,
            load_if_exists=True,  # Load the study if it already exists
        )

        
        # Optimize the objective function
        study.optimize(objective, n_trials=args.bayesopt_step_num)

        # Retrieve all Pareto-optimal trials
        best_trials = study.best_trials

        print("Pareto-optimal trials and their hyperparameters:")
        for i, trial in enumerate(best_trials):
            print(f"Trial {i}:")
            print(f"  Objective Values: {trial.values}")  # Objective values for the trial
            print(f"  Hyperparameters: {trial.params}")  # Hyperparameters for the trial


        # Save Pareto-optimal trials to a JSON file
        pareto_results = [
            {"objective_values": trial.values, "params": trial.params} for trial in best_trials
        ]
        save_path = f"{BASE_STORAGE}/best_hparams_{args.run_name}_run{args.run_no}_poolby{args.pooling_method}_{args.PLM}.json"
        with open(save_path, "w") as f:
            import json
            json.dump(pareto_results, f, indent=4)

        print(f"Pareto-optimal results saved to {save_path}")



