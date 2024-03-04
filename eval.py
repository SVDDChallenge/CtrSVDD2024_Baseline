import argparse
import os, sys
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import SVDD2024
from models.model import SVDDModel
from utils import seed_worker, set_seed

def main(args):
    # Set the seed for reproducibility
    set_seed(args.random_seed)
    
    path = args.base_dir
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    test_dataset = SVDD2024(path, partition="test")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker)
    
    # Create the model
    model = SVDDModel(frontend=args.encoder, device=device).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    scores_out = args.output_path

    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Testing")):
            x, label, filenames = batch
            x = x.to(device)
            label = label.to(device)
            _, pred = model(x)
            for p, filename in zip(pred, filenames):
                with open(os.path.join(scores_out, f'scores_{args.encoder}_randomseed_{args.random_seed}.txt'), "a") as f:
                    f.write(f"{filename} {p.item()}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", type=int, default=42, help="The random seed.")
    parser.add_argument("--base_dir", type=str, required=True, help="The base directory of the dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="The path to the model.")
    parser.add_argument("--gpu", type=int, default=0, help="The GPU to use.")
    parser.add_argument("--encoder", type=str, default="mfcc", help="The encoder to use.")
    parser.add_argument("--batch_size", type=int, default=36, help="The batch size for training.")
    parser.add_argument("--num_workers", type=int, default=12, help="The number of workers for the data loader.")
    parser.add_argument("--output_path", type=str, default="scores", help="The output folder for the scores.")
    
    args = parser.parse_args()
    main(args)