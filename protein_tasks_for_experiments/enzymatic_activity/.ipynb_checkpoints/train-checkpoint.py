import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from data import get_loader
from models import EnzymePredictor
from utils import set_seed, parse_args, setup_logging
import wandb
import csv
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, matthews_corrcoef
import os

def train(model, train_loader, optimizer, scheduler, device, class_weights, args):
    model.train()
    total_loss = 0

    if args.rapid_debug_mode:
        batch_counter = 0
        max_batch = 50

    # Ensure class weights are on the correct device
    class_weights = class_weights.to(device)  # Move class weights to the correct device
    
    for data, target in train_loader:
        if args.rapid_debug_mode:
            batch_counter += 1
            if batch_counter > max_batch:
                break
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).to(device)
        loss = F.binary_cross_entropy_with_logits(output, target, weight=class_weights[target.long()].to(device))
        loss.backward()

        # Calculate and log the non-clipped gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
        if not args.rapid_debug_mode:
            pass
            #wandb.log({"gradient_norm": total_norm.item()})
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
        optimizer.step()
        if not scheduler is None:
            scheduler.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, test_loader, device, args, threshold=0.5):
    model.eval()
    total_preds, total_targets = [], []
    predictions = []
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            probs = torch.sigmoid(outputs).cpu()  # Convert logits to probabilities
            predicted_labels = (probs > threshold).int()  # Apply threshold
            predictions.extend(zip(probs.tolist(), predicted_labels.tolist(), targets.cpu().tolist()))
            total_preds.extend(probs.squeeze().tolist())  # Flatten the probabilities for AUROC
            total_targets.extend(targets.cpu().tolist())

    # Calculate metrics
    auroc = roc_auc_score(total_targets, total_preds)
    auprc = average_precision_score(total_targets, total_preds)
    f1 = f1_score(total_targets, [int(p > threshold) for p in total_preds])
    mcc = matthews_corrcoef(total_targets, [int(p > threshold) for p in total_preds])

    return predictions, auroc, auprc, f1, mcc

def validate(model, val_loader, device, class_weights, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.binary_cross_entropy_with_logits(output, target)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    args = parse_args()
    set_seed(args.seed)
    logging = setup_logging(args)
    
    train_loader = get_loader(args.train_path, args.pooling_method, args.batch_size, shuffle=True,
                              exponent_class_weight = args.exponent_class_weight,
                             rapid_debug_mode = args.rapid_debug_mode)
    val_loader = get_loader(args.val_path, args.pooling_method, args.batch_size, 
                            shuffle=False,
                              exponent_class_weight = args.exponent_class_weight,
                             rapid_debug_mode = args.rapid_debug_mode)
    
    test_loader_human = get_loader(args.test_path_human, args.pooling_method, args.batch_size, 
                             shuffle=False,
                             exponent_class_weight = args.exponent_class_weight,
                             rapid_debug_mode = args.rapid_debug_mode)

    test_loader_mammalian = get_loader(args.test_path_mammalian, args.pooling_method, args.batch_size, 
                             shuffle=False,
                             exponent_class_weight = args.exponent_class_weight,
                             rapid_debug_mode = args.rapid_debug_mode)

    test_loader_vertebrate = get_loader(args.test_path_vertebrate, args.pooling_method, args.batch_size, 
                             shuffle=False,
                             exponent_class_weight = args.exponent_class_weight,
                             rapid_debug_mode = args.rapid_debug_mode)

    test_loader_animal = get_loader(args.test_path_animal, args.pooling_method, args.batch_size, 
                             shuffle=False,
                             exponent_class_weight = args.exponent_class_weight,
                             rapid_debug_mode = args.rapid_debug_mode)

    test_loader_eukaryote = get_loader(args.test_path_eukaryote, args.pooling_method, args.batch_size, 
                             shuffle=False,
                             exponent_class_weight = args.exponent_class_weight,
                             rapid_debug_mode = args.rapid_debug_mode)


    

    
    class_weights = train_loader.dataset.class_weights
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}", flush=True)
    model = EnzymePredictor(1280, args.hidden_dim, 1, dropout_rate=args.dropout_rate, negative_slope=args.slopeLeakyReLU).to(device)
    
    if args.early_stopping:
        # Replace OneCycleLR with a scheduler more compatible with early stopping
        optimizer = optim.AdamW(model.parameters(), lr=args.max_LR, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                               factor=args.reduce_lr_ratio, 
                                                               patience=args.patience_lr, 
                                                               verbose=True)
        print("initialized ReduceLROnPlateau scheduler", flush=True)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = OneCycleLR(optimizer, max_lr=args.max_LR, steps_per_epoch=len(train_loader), epochs=args.num_epochs,
                              pct_start = args.percent_epochs_lr_rise)
        print("initialized OneCycleLR scheduler", flush=True)

    best_val_loss = float('inf')
    epochs_since_improvement = 0  # Track the number of epochs since last improvement for early stopping
    os.makedirs(f"best_models/{args.pooling_method}", exist_ok=True)

    
    best_model_path = os.path.join(f"best_models/{args.pooling_method}", f'model-run{args.run_no}.pt')

    if args.other_notes is None:
        best_model_path = os.path.join(f"best_models/{args.pooling_method}", f'model-run{args.run_no}.pt')
    else:
        best_model_path = os.path.join(f"best_models/{args.pooling_method}", f'model-run{args.run_no}_{args.other_notes}.pt')
        #file_name = f"{args.run_no}_{args.other_notes}_log.txt"

    epochs = args.num_epochs
    
    if not args.eval_only:
        if not args.rapid_debug_mode:
            pass
            #wandb.init(project="poolingEnzyme", name= f"{args.pooling_method}-run{args.run_no}", config=vars(args))
        
        for epoch in range(epochs):
            if args.early_stopping: # matter of whether you pass the scheduler or not
                train_loss = train(model, train_loader, optimizer=optimizer, scheduler=None, device=device, class_weights=class_weights, args=args)
            else:
                train_loss = train(model, train_loader, optimizer=optimizer, scheduler=scheduler, device=device, class_weights=class_weights, args=args)
            val_loss = validate(model, val_loader, device, class_weights, args)

            if args.early_stopping:
                scheduler.step(val_loss)


            current_lr = optimizer.param_groups[0]['lr']
            
            if not args.rapid_debug_mode:
                logging.info(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
            if not args.rapid_debug_mode:
                pass
                commment = """
                wandb.log({"train_loss": train_loss, 
                           "val_loss": val_loss,
                           "current_lr": current_lr,
                          "epoch": epoch})"""

            # Save the model if it has the best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
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
    model.load_state_dict(torch.load(best_model_path))

    tests = ["human", "mammalian", "vertebrate", "animal", "eukaryote"]
    test_loaders = [test_loader_human, test_loader_mammalian, test_loader_vertebrate, test_loader_animal, test_loader_eukaryote]

    for i in range(len(tests)):
        group = tests[i]
        test_loader = test_loaders[i]
    

        # Testing and logging results
        test_predictions, auroc, auprc, f1, mcc = test(model, test_loader, device, args)
        if args.other_notes is None:
            prediction_file = f"run{args.run_no}_pred_{group}.csv"
        else:
            prediction_file = f"run{args.run_no}_{args.other_notes}_pred_{group}.csv"

        dir_pred = f"predictions/{args.pooling_method}"
        os.makedirs(dir_pred, exist_ok=True)
        
        with open(f'{dir_pred}/{prediction_file}', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Probability', 'Predicted Label', 'Actual Label'])
            writer.writerows(test_predictions)
        
        # Log metrics to console and wandb
        metrics = {
            f"test_auroc_{group}": auroc,
            f"test_auprc_{group}": auprc,
            f"test_f1_score_{group}": f1,
            f"test_mcc_{group}": mcc
        }
        if not args.rapid_debug_mode:
            logging.info(f'Test Metrics_{group}: {metrics}\n\n')
            #wandb.log(metrics)

if __name__ == "__main__":
    main()



