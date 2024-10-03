import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from data_efficient import get_loader, preload_embeddings
from models import InteractionPredictor
from utils_efficient import set_seed, parse_args, setup_logging
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
    if args.rapid_debug_mode:
        print(f"number of batches is {len(train_loader)} with minibatch size of {args.batch_size}")
    
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

    embedding_dict = preload_embeddings(args.pooling_method, rapid_debug_mode=args.rapid_debug_mode)
    
    train_loader = get_loader(args.train_pos_path, args.train_neg_path, args.pooling_method, args.batch_size, shuffle=True,
                              exponent_class_weight = args.exponent_class_weight,
                             rapid_debug_mode = args.rapid_debug_mode,
                             embedding_dict=embedding_dict)
    val_loader = get_loader(args.val_pos_path, args.val_neg_path, args.pooling_method, args.batch_size, shuffle=False,
                              exponent_class_weight = args.exponent_class_weight,
                             rapid_debug_mode = args.rapid_debug_mode,
                           embedding_dict=embedding_dict)
    test_loader = get_loader(args.test_pos_path, args.test_neg_path, args.pooling_method, args.batch_size, shuffle=False,
                              exponent_class_weight = args.exponent_class_weight,
                             rapid_debug_mode = args.rapid_debug_mode,
                            embedding_dict=embedding_dict)
    class_weights = train_loader.dataset.class_weights
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}", flush=True)
    model = InteractionPredictor(2 * 1280, args.hidden_dim, 1, dropout_rate = args.dropout_rate,
                                negative_slope = args.slope_Leaky_ReLU).to(device)
    
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
    os.makedirs(f"best_models_efficient/{args.pooling_method}", exist_ok=True)
    if args.other_notes is None:
        best_model_path = os.path.join(f'best_models_efficient/{args.pooling_method}', f'{args.run_no}_model.pt')
    else:
        best_model_path = os.path.join(f'best_models_efficient/{args.pooling_method}', f'{args.run_no}_{args.other_notes}_model.pt')

    epochs = args.num_epochs

    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    if not args.eval_only:
        if not args.rapid_debug_mode:
            pass
            #wandb.init(project="poolingPPI", name= f"{args.pooling_method}-run{args.run_no}", config=vars(args))
        
        for epoch in range(epochs):
            if args.early_stopping: # matter of whether you pass the scheduler or not
                train_loss = train(model, train_loader, optimizer=optimizer, scheduler=None, device=device, class_weights=class_weights, args=args)
            else:
                train_loss = train(model, train_loader, optimizer=optimizer, scheduler=scheduler, device=device, class_weights=class_weights, args=args)
            val_loss = validate(model, val_loader, device, class_weights, args)

            if args.early_stopping:
                scheduler.step(val_loss)


            current_lr = optimizer.param_groups[0]['lr']

            if args.rapid_debug_mode:
                print(f"train_loss {train_loss}", flush=True)
                print(f"val_loss {val_loss}", flush=True)
            
            if not args.rapid_debug_mode:
                logging.info(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')
            if not args.rapid_debug_mode:
                pass
                comment = """
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

    # Testing and logging results
    test_predictions, auroc, auprc, f1, mcc = test(model, test_loader, device, args)
    pred_dir = f"./predictions_efficient/{args.pooling_method}"
    os.makedirs(pred_dir, exist_ok=True)
    if args.other_notes is None:
        prediction_file = f"{pred_dir}/{args.run_no}_pred.csv"
    else:
        prediction_file = f"{pred_dir}/{args.run_no}_pred_{args.other_notes}_pred.csv"
    
    with open(f'{prediction_file}', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Probability', 'Predicted Label', 'Actual Label'])
        writer.writerows(test_predictions)

    # Log metrics to console and wandb
    metrics = {
        "test_auroc": auroc,
        "test_auprc": auprc,
        "test_f1_score": f1,
        "test_mcc": mcc
    }
    if not args.rapid_debug_mode:
        logging.info(f'Test Metrics: {metrics}')
        #wandb.log(metrics)

if __name__ == "__main__":
    main()



