import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score, f1_score, roc_auc_score, multilabel_confusion_matrix, average_precision_score
from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import wandb
import argparse
import torch
import os
import datetime
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau

from SubcellularLocalizationDataset import SubcellularLocalizationDataset, preload_embeddings
from model import MultiLabelNN
from utils_localization import custom_none_collate_fn


label_map = {1: "Cell membrane",            
             2: "Cytoplasm",
             3: "Endoplasmic reticulum",
             4: "Golgi apparatus",
             5: "Lysosome/Vacuole", 
             6: "Mitochondrion",
             7: "Nucleus",
             8: "Peroxisome"
             }

def log_to_file(filename, content):
    with open(filename, "a") as file:
        # Convert numpy array to string if content is an array
        if isinstance(content, np.ndarray):
            content = np.array2string(content)
        file.write(content + "\n")

def evaluate_model(data_loader, model, criterion, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []  # Initialize a list to store the probabilities
    total_loss = 0.0

    with torch.no_grad():
        for embeddings, labels in data_loader:
            embeddings, labels = embeddings.float().to(device), labels.float().to(device)  # Convert embeddings and labels to float
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * embeddings.size(0)

            # Store the probabilities before applying the threshold
            # If your model outputs raw logits, you may need to apply a sigmoid function to convert them to probabilities
            probs = torch.sigmoid(outputs)  # Apply sigmoid if outputs are logits
            all_probs.append(probs.cpu())
            
            # Convert outputs to a NumPy array
            outputs_np = outputs.cpu().numpy()
            
            # Now you can safely use NumPy functions on outputs_np
            threshold = min(np.mean(outputs_np) + 3*np.std(outputs_np), 0.99*np.max(outputs_np), 0.5)
            #threshold = 0.5
            
            # Convert threshold back to tensor to perform comparison on GPU if needed
            threshold_tensor = torch.tensor(threshold, device=device)
            
            preds = outputs >= threshold_tensor  # Use the threshold_tensor for comparison
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Convert lists of tensors to tensors
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()  # Convert all_probs to a tensor and then to a NumPy array

    avg_loss = total_loss / len(data_loader.dataset)

    # Micro and Macro F1 scores
    f1_micro = f1_score(all_labels, all_preds, average='micro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    # Hamming Loss
    h_loss = hamming_loss(all_labels, all_preds)

    

    # Hamming Accuracy (1 - Hamming Loss)
    h_accuracy = 1 - h_loss

    # Multi-label MCC
    mcc_scores = [pearsonr(all_preds[:, i], all_labels[:, i])[0] for i in range(all_preds.shape[1])]
    mcc_scores = [x for x in mcc_scores if not np.isnan(x)]  # Remove NaN values which might occur with constant labels/predictions
    mcc = np.mean(mcc_scores)

    return avg_loss, f1_micro, f1_macro, mcc, h_loss, h_accuracy, all_preds, all_labels, all_probs

def evaluate_individual_labels(all_labels, all_preds, all_probs):
    num_labels = all_labels.shape[1]
    metrics = {
        'label': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'auprc': [],
    }

    

    # Iterate over each label
    for i in range(num_labels):
        #label = f'Label {i+1}'
        label = label_map[i+1]
        metrics['label'].append(label)

        # Precision, Recall, and F1 Score for each label
        precision = precision_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        recall = recall_score(all_labels[:, i], all_preds[:, i], zero_division=0)
        f1 = f1_score(all_labels[:, i], all_preds[:, i], zero_division=0)

        metrics['precision'].append(precision)
        metrics['recall'].append(recall)
        metrics['f1_score'].append(f1)

        # ROC-AUC Score for each label
        # Note: ROC-AUC can be undefined for labels with only one class present in `y_true`
        # You might need to handle such cases depending on your data
        if len(np.unique(all_labels[:, i])) > 1 and not args.rapid_debug_mode:
            auprc = average_precision_score(all_labels[:, i], all_probs[:, i])
            metrics['auprc'].append(auprc)
            wandb.log({f"{label}_auprc": auprc})
        else:
            metrics['auprc'].append('Undefined')

    metrics["label"].append("global")
    metrics['precision'].append("not-calculated")
    metrics['recall'].append("not-calculated")
    metrics['f1_score'].append("not-calculated")

    flat_all_labels = all_labels.ravel()
    flat_all_probs = all_probs.ravel()
    # Calculate the global AUPRC
    global_auprc = average_precision_score(flat_all_labels, flat_all_probs)
    metrics['auprc'].append(global_auprc)
    if not args.rapid_debug_mode:
        wandb.log({
            "global_auprc": global_auprc
        })
    

    return pd.DataFrame(metrics)

def safe_pearsonr(x, y):
    # Check if either input array is empty
    if len(x) == 0 or len(y) == 0:
        return np.nan  # Return NaN for empty arrays
    # Check if either input array is constant
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return pearsonr(x, y)[0]


def main():
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("test set address is ", args.path_to_test_csv)
    seed_value = 42  # Set a specific seed value 
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    # Ensure CUDA reproducibility (if you're using a GPU)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    

    if not args.rapid_debug_mode:
        if not args.eval_only:
            pass
            # wandb.init(project="pooling", name=f"{args.path_to_embeddings}-{args.run_no}", config=vars(args))
        else:
            pass
            #wandb_name = f"{args.run_name}_eval_test_only"
            #wandb.init(project="pooling", name=f"{args.path_to_embeddings}-{args.run_no}", config=vars(args))

    
    if not args.eval_only:
        # Load datasets
        df_loc_trainval_metazoa = pd.read_csv(args.path_to_trainval_csv)
        
    
        print("Loaded the dataframes without error")
    
        #df_train, df_val = train_test_split(df_loc_trainval_metazoa, test_size=0.1, random_state=42)
        # a better way to split train and val sets
        # Assuming df_loc_trainval_metazoa['labels'] contains the multi-labels as a list of lists or a multi-label binary array
        #labels = np.array(df_loc_trainval_metazoa['labels'].tolist())  # Convert labels column to a NumPy array if it's not already
        label_columns = list(df_loc_trainval_metazoa.columns)[1:]
        labels = df_loc_trainval_metazoa[label_columns].values
        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)

        ####### DATA BALANCING #########
        # Calculate the positive cases for each class
        class_positives = labels.sum(axis=0)
        # Calculate class weights inversely proportional to class frequencies
        total_labels = labels.shape[0]
        class_weights = total_labels / class_positives

        if args.rapid_debug_mode:
            print(f"class_positives {class_positives}", flush=True)
            print(f"total_labels {total_labels}", flush=True)
            print(f"len(label_columns) {len(label_columns)}", flush=True)

        
        # Applying power to scale class weights
        class_weights = np.power(class_weights, args.power_for_balance_penalty_taming)
        
        # Clamping the class weights between 0.9 and 3
        class_weights = np.clip(class_weights, args.minClassWeight, args.maxClassWeight)
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

        if args.rapid_debug_mode:
            print(f"class_weights after scaling{class_weights}", flush=True)
    
        for train_index, val_index in msss.split(df_loc_trainval_metazoa, labels):
            df_train = df_loc_trainval_metazoa.iloc[train_index]
            df_val = df_loc_trainval_metazoa.iloc[val_index]

        if os.path.exists(f"dict_embeddings/{args.path_to_embeddings}.pt"):
            embedding_dict = torch.load(f"dict_embeddings/{args.path_to_embeddings}.pt")
            print("loaded the existing dictionary of embeddings")
        else:
            embedding_dict = preload_embeddings(args.path_to_embeddings, rapid_debug_mode=args.rapid_debug_mode)
    
        
        print("Split the dataset iteratively")
    
        train_loader = DataLoader(SubcellularLocalizationDataset(df_train, embedding_dict), batch_size=64, shuffle=True,
                                 collate_fn=custom_none_collate_fn)
        val_loader = DataLoader(SubcellularLocalizationDataset(df_val, embedding_dict), batch_size=64, shuffle=False,
                               collate_fn=custom_none_collate_fn)
    
    print("test set address is ", args.path_to_test_csv)
    df_loc_test = pd.read_csv(args.path_to_test_csv)
    test_loader = DataLoader(SubcellularLocalizationDataset(df_loc_test, embedding_dict), batch_size=64, shuffle=False,
                            collate_fn=custom_none_collate_fn)

    print("Created dataloaders")

    # Initialize model, criterion, optimizer
    model = MultiLabelNN(input_size=1280, num_classes=8, dropout = args.dropout, slope_Leaky_ReLU = args.slope_Leaky_ReLU).to(device)  # Adjust the input_size and num_classes as needed
    #criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
    #optimizer = Adam(model.parameters(), lr=0.001)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)

    
    num_epochs = args.epochs
    patience = max(args.patience, args.epochs//20)

    if args.rapid_debug_mode:
        num_epochs = 1
        patience = 1

    best_val_loss = float('inf')
    best_mcc = 0
    patience_counter = 0  # Early Stopping Counter
    print("got to before the first epoch")


    if args.rapid_debug_mode:
        MAX_BATCHES = 5
        batch_counter = 0

    # train loop
    os.makedirs(f'logs_efficient/best_model_checkpoints/{args.path_to_embeddings}', exist_ok=True)
    checkpoint_path = f'logs_efficient/best_model_checkpoints/{args.path_to_embeddings}/{args.run_no}.pth'
    if not args.eval_only:
        for epoch in range(num_epochs):
            print(f"starting epoch {epoch}")
            model.train()
            total_loss = 0.0
            for embeddings, labels in train_loader:
                if args.rapid_debug_mode:
                    batch_counter += 1
                    if batch_counter > MAX_BATCHES:
                        break
                embeddings, labels = embeddings.float().to(device), labels.float().to(device)  # Convert embeddings and labels to float  
                # Send data to device
                optimizer.zero_grad()
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                loss.backward()
                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
                optimizer.step()
                total_loss += loss.item() * embeddings.size(0)
    
            avg_train_loss = total_loss / len(df_train)
            os.makedirs(f'logs_efficient/training_logs/{args.path_to_embeddings}', exist_ok=True)
            if not args.rapid_debug_mode:
                pass
                #wandb.log({"epoch": epoch, "train_loss": avg_train_loss})
            log_to_file(f"logs_efficient/training_logs/{args.path_to_embeddings}/{args.run_no}.txt", f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}')
    
            if not args.rapid_debug_mode:
                # Evaluation using the updated evaluate_model function
                val_loss, f1_micro, f1_macro, mcc, h_loss, h_accuracy, _, _, _ = evaluate_model(val_loader, model, criterion, device)

                scheduler.step(val_loss)
                # Logging metrics to wandb
                comment = """
                wandb.log({
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "f1_micro": f1_micro,
                    "f1_macro": f1_macro,
                    "mcc": mcc,
                    "hamming_loss": h_loss,
                    "hamming_accuracy": h_accuracy
                })"""
                
                # Printing out the metrics for this epoch
                print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, '
                      f'F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, MCC: {mcc:.4f}, '
                      f'Hamming Loss: {h_loss:.4f}, Hamming Accuracy: {h_accuracy:.4f}', flush=True)
                
                # Checkpoint Saving Logic based on validation loss improvement
                if val_loss < best_val_loss or mcc > best_mcc:
                    best_val_loss = val_loss
                    best_mcc = mcc
                    patience_counter = 0  # Reset patience counter if improvement
                    # Construct the checkpoint file name based on embeddings path and run number
                    checkpoint_filename = checkpoint_path
                    torch.save(model.state_dict(), checkpoint_filename)
                    print(f"Checkpoint saved at epoch {epoch+1} with validation loss: {val_loss:.4f}", flush=True)
                else:
                    patience_counter += 1  # Increment patience counter if no improvement
                
                if patience_counter >= patience:
                    print(f"Stopping early at epoch {epoch+1} due to no improvement in validation loss for {patience} epochs.")
                    break


    # Load the best model from checkpoint for evaluation
    os.makedirs(f'logs_efficient/best_model_checkpoints/{args.path_to_embeddings}', exist_ok=True)
    checkpoint_path = f'logs_efficient/best_model_checkpoints/{args.path_to_embeddings}/{args.run_no}.pth'
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        print("Loaded best model from checkpoint for evaluation.", flush=True)
    else:
        if not args.eval_only:
            print("Checkpoint not found. Evaluating with current model state.")
        else:
            raise Exception("Checkpoint not found for the eval only mode. The model was not loaded.")


    
    # Evaluate the model on the test set
    test_loss, f1_micro, f1_macro, mcc, h_loss, h_accuracy, all_preds, all_labels, all_probs = evaluate_model(test_loader, model, criterion, device)

    mcm = multilabel_confusion_matrix(all_labels, all_preds)
    print("Multi-label Confusion Matrix:\n", mcm)
    os.makedirs(f'logs_efficient/confusion/{args.path_to_embeddings}', exist_ok=True)
    
    log_to_file(f"logs_efficient/confusion/{args.path_to_embeddings}/{args.run_no}.txt",
               mcm)
    log_to_file(f"logs_efficient/confusion/{args.path_to_embeddings}/{args.run_no}.txt", 
                f"\n\nclass weights are\n{class_weights}")
    log_to_file(f"logs_efficient/confusion/{args.path_to_embeddings}/{args.run_no}.txt", 
                f"\n\arguments are are\n")

    for arg, value in vars(args).items():
        if isinstance(value, str):  # Check if the argument value is a string
            # Replace <your-username> with "alptartici" in the string
            new_value = value.replace("<your-username>", "alptartici")
            # Set the modified value back to the args namespace
            setattr(args, arg, new_value)
        print(arg, value, flush=True)
        log_to_file(f"logs_efficient/confusion/{args.path_to_embeddings}/{args.run_no}.txt", 
                f"{arg}\t{value}\n")
        

    metrics_df_auprc = evaluate_individual_labels(np.array(all_labels), np.array(all_preds), np.array(all_probs))

    os.makedirs(f'logs_efficient/metrics/{args.path_to_embeddings}', exist_ok=True)
    metrics_df_auprc.to_csv(f'logs_efficient/metrics/{args.path_to_embeddings}/{args.run_no}.csv', index=False)
    
    # Log metrics to wandb
    if not args.rapid_debug_mode:
        pass 
        comment = """
        wandb.log({
            "test_loss": test_loss,
            "test_f1_micro": f1_micro,
            "test_f1_macro": f1_macro,
            "test_mcc": mcc,
            "test_hamming_loss": h_loss,
            "test_hamming_accuracy": h_accuracy
        })"""
    
    # Print out the metrics for the test set evaluation
    print(f'Test Set Evaluation - Loss: {test_loss:.4f}, F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, '
          f'MCC: {mcc:.4f}, Hamming Loss: {h_loss:.4f}, Hamming Accuracy: {h_accuracy:.4f}')
    
    # Log metrics to a file
    os.makedirs(f'logs_efficient/test_logs/{args.path_to_embeddings}', exist_ok=True)
    log_to_file(f"logs_efficient/test_logs/{args.path_to_embeddings}/{args.run_no}.txt", 
                f'Test Set Evaluation - Loss: {test_loss:.4f}, F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}, '
                f'MCC: {mcc:.4f}, Hamming Loss: {h_loss:.4f}, Hamming Accuracy: {h_accuracy:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model for subcellular location prediction.')
    # HYPERPARAMS
    parser.add_argument('--hyperparam_group', type=int, default=-1, help='hyperparam group')
    parser.add_argument('--path_to_embeddings', type=str,  help='Path to the embeddings directory')
    parser.add_argument('--path_to_trainval_csv', type=str, 
                        default = "./datasets/df_loc_trainval_seqlen_1022_dp_14503.csv", help='Path to the trainval CSV file')
    parser.add_argument('--path_to_test_csv', type=str, 
                        default = "./datasets/df_loc_test_seqlen_550_dp_1382.csv", help='Path to the test CSV file')
    parser.add_argument('--run_no', type=int, required=True, help='the index of run')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train the model (default: 500)')
    parser.add_argument('--run_name', type=str, default=datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
                        help='run name for logging')
    parser.add_argument('--eval_only', action='store_true', help='Run evaluation on test set only, without training')
    parser.add_argument('--rapid_debug_mode', action='store_true', help='quick debug surgery')
    parser.add_argument('--patience', type=int, default=50, help='Number of epochs to to have patience for (default: 50)')
    parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate in the models')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay in AdamW for regularization')
    parser.add_argument('--power_for_balance_penalty_taming', type=float, default=1.0, help='take this power of class weights')
    parser.add_argument('--minClassWeight', type=float, default=0.001, help='minimum class weight for clamping')
    parser.add_argument('--maxClassWeight', type=float, default=10000, help='max class weight for clamping')


    parser.add_argument('--slope_Leaky_ReLU', type=float, default=0.01, help='leakyReluSlope')


    args = parser.parse_args()
    
    if args.hyperparam_group > -0.5:
        # Read hyperparameters from CSV
        df_params = pd.read_csv('hyperparams_final.csv', index_col='hyperparam_group')
        
        if args.hyperparam_group in df_params.index:
            # Update the arguments with the parameters from the CSV
            group_params = df_params.loc[args.hyperparam_group]
            args.lr = float(group_params['lr'])
            args.weight_decay = float(group_params['weight_decay'])
            args.dropout = float(group_params['dropout'])
            args.slope_Leaky_ReLU = float(group_params['slope_Leaky_ReLU'])
            args.power_for_balance_penalty_taming = float(group_params['power_for_balance_penalty_taming'])
            
        else:
            print(f"No hyperparameter group found for {args.hyperparam_group}. Using defaults.", flush=True)   

    
    

    # run command:
    # python -u train.py --path_to_embeddings max_pooled --run_no 0 --epochs 1 --run_name trial 2>&1 | tee output.txt
    
    # Iterate over all arguments, replace <your-username> with "alptartici" where necessary
    for arg, value in vars(args).items():
        print(arg, value, flush=True)
    main()
