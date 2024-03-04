import argparse
import os, sys
import torch
import numpy as np
from tqdm import tqdm
import datetime, random
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datasets import SVDD2024
from models.model import SVDDModel
from utils import seed_worker, set_seed, compute_eer

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, use_logits=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.use_logits = use_logits
        
    def forward(self, logits, targets):
        if self.use_logits:
            bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        else:
            bce_loss = F.binary_cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

def main(args):
    # Set the seed for reproducibility
    set_seed(42)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Create the dataset
    path = args.base_dir
    train_dataset = SVDD2024(path, partition="train")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker)
    
    dev_dataset = SVDD2024(path, partition="dev")
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker)
    
    # Create the model
    model = SVDDModel(device, frontend=args.encoder).to(device)

    # Create the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-9, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    # Create the directory for the logs
    log_dir = os.path.join(args.log_dir, args.encoder)
    os.makedirs(log_dir, exist_ok=True)
    
    # get current time
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, current_time)
    os.makedirs(log_dir, exist_ok=True)

    # Create the summary writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Create the directory for the checkpoints
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config for reproducibility
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        f.write(str(vars(args)))
        
    criterion = BinaryFocalLoss()
    
    best_val_eer = 1.0

    # Train the model
    for epoch in range(args.epochs):
        model.train()
        pos_samples, neg_samples = [], []
        for i, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")):
            if args.debug and i > 20:
                break
            x, label, _ = batch
            x = x.to(device)
            label = label.to(device)
            soft_label = label.float() * 0.9 + 0.05
            _, pred = model(x)
            loss = criterion(pred, soft_label.unsqueeze(1))
            pos_samples.append(pred[label == 1].detach().cpu().numpy())
            neg_samples.append(pred[label == 0].detach().cpu().numpy())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            writer.add_scalar("Loss/train", loss.item(), epoch * len(train_loader) + i)
        scheduler.step()
        writer.add_scalar("LR/train", scheduler.get_last_lr()[0], epoch * len(train_loader) + i)
        writer.add_scalar("EER/train", compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0], epoch)
        
        
        model.eval()
        val_loss = 0
        pos_samples, neg_samples = [], []
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dev_loader, desc=f"Validation")):
                if args.debug and i > 20:
                    break
                x, label, _ = batch
                x = x.to(device)
                label = label.to(device)
                _, pred = model(x)
                soft_label = label.float() * 0.9 + 0.05
                loss = criterion(pred, soft_label.unsqueeze(1))
                pos_samples.append(pred[label == 1].detach().cpu().numpy())
                neg_samples.append(pred[label == 0].detach().cpu().numpy())
                val_loss += loss.item()
            val_loss /= len(dev_loader)
            val_eer = compute_eer(np.concatenate(pos_samples), np.concatenate(neg_samples))[0]
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("EER/val", val_eer, epoch)
            if val_eer < best_val_eer:
                best_val_eer = val_eer
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"best_model.pt"))
            if epoch % 10 == 0: # Save every 10 epochs
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{epoch}_EER_{val_eer}.pt"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset.")
    parser.add_argument("--epochs", type=int, default=100, help="The number of epochs to train.")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode.")
    parser.add_argument("--gpu", type=int, default=0, help="The GPU to use.")
    parser.add_argument("--encoder", type=str, default="rawnet", help="The encoder to use.")
    parser.add_argument("--batch_size", type=int, default=24, help="The batch size for training.")
    parser.add_argument("--num_workers", type=int, default=6, help="The number of workers for the data loader.")
    parser.add_argument("--log_dir", type=str, default="logs", help="The directory for the logs.")
    
    args = parser.parse_args()
    main(args)