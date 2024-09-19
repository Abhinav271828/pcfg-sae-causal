from torch.utils.data import DataLoader

from .model import *
from .data import *

import wandb
from tqdm import tqdm
import json

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data = SAEData(args.model_dir, args.ckpt, args.layer_name, config=args.config, device=device)
    val_data = SAEData(args.model_dir, args.ckpt, args.layer_name, config=args.config, device=device)
    embedding_size = train_data[0][0].size(-1)
    model = SAE(embedding_size, args.exp_factor * embedding_size, pre_bias=args.pre_bias, k=args.k, sparsemax=args.sparsemax, norm=args.norm).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    criterion_caus = nn.KLDivLoss(reduction='batchmean')

    # Train the SAE
    wandb.init(project="pcfg-sae-causal")
    wandb.run.name = wandb.run.id
    wandb.run.save()
    wandb.config.update(vars(args))

    for epoch in range(1):
        prev_loss = float('inf')
        loss_increasing = 0

        train_it = 0
        for activation, logits, seq, in tqdm(train_data, desc="Training", total=args.train_iters):
            if train_it > args.train_iters: break

            activation = activation.to(device)
            optimizer.zero_grad()
            if 'input' in args.norm:
                norm = torch.norm(activation, p=2, dim=-1)
                activation = activation / norm.unsqueeze(-1)
            latent, recon = model(activation)

            recon_loss = criterion(recon, activation)

            reg_loss = torch.norm(latent, p=1) if args.alpha else 0

            # Find random nonzero elements for each row
            nonzero_mask = (latent != 0)
            nonzero_counts = nonzero_mask.sum(-1)  # For each row, the number of nonzero elements
            nonzero_indices = torch.arange(latent.size(-1)).expand_as(latent)
            nonzero_indices_per_row = torch.where(nonzero_mask, nonzero_indices, torch.tensor(-1)) # Same shape as latent; -1 at zero positions and column number elsewhere
            sorted_nonzero_indices_per_row = torch.argsort(nonzero_indices_per_row, dim=1, descending=True)
            # Now the first k elements are indices of (indices of) nonzero elements, and the rest are indices of 0
            # k depends on the row; it is the number of nonzero elements in that row
            indices_of_indices = (torch.rand(latent.size(0)) * nonzero_counts).long() # Randomly select a nonzero index for each row
            selected_indices = sorted_nonzero_indices_per_row[torch.arange(latent.size(0)), indices_of_indices]

            # Ablate selected indices to zero
            abl_latent = latent.clone()
            abl_latent[torch.arange(latent.size(0)), selected_indices] = 0
            ablated_recon = model.decoder(abl_latent)

            # Compare the outputs
            pure_logits = train_data.intervene(seq, recon)
            abl_logits = train_data.intervene(seq, ablated_recon)
            causal_loss = -criterion_caus(F.log_softmax(abl_logits, dim=-1), F.softmax(pure_logits, dim=-1))

            loss = recon_loss + causal_loss * args.beta
            loss += reg_loss * args.alpha if args.alpha else 0

            loss.backward()
            optimizer.step()
            train_loss = loss.item()

            if args.patience and train_loss > prev_loss:
                loss_increasing += 1
                if loss_increasing == args.patience: break
            else:
                loss_increasing = 0
                prev_loss = train_loss

            if train_it % args.val_interval == 0:
                model.eval()
                val_loss = 0
                val_it = 0
                for activation, logits, seq in val_data:
                    if val_it > args.val_iters: break
                    activation = activation.to(device)
                    if 'input' in args.norm:
                        norm = torch.norm(activation, p=2, dim=-1)
                        activation = activation / norm.unsqueeze(-1)
                    latent, recon = model(activation)

                    recon_loss = criterion(recon, activation)

                    # Find random nonzero elements for each row
                    nonzero_mask = (latent != 0)
                    nonzero_counts = nonzero_mask.sum(-1)  # For each row, the number of nonzero elements
                    nonzero_indices = torch.arange(latent.size(-1)).expand_as(latent)
                    nonzero_indices_per_row = torch.where(nonzero_mask, nonzero_indices, torch.tensor(-1)) # Same shape as latent; -1 at zero positions and column number elsewhere
                    sorted_nonzero_indices_per_row = torch.argsort(nonzero_indices_per_row, dim=1, descending=True)
                    # Now the first k elements are indices of (indices of) nonzero elements, and the rest are indices of 0
                    # k depends on the row; it is the number of nonzero elements in that row
                    indices_of_indices = (torch.rand(latent.size(0)) * nonzero_counts).long() # Randomly select a nonzero index for each row
                    selected_indices = sorted_nonzero_indices_per_row[torch.arange(latent.size(0)), indices_of_indices]

                    # Ablate selected indices to zero
                    abl_latent = latent.clone()
                    abl_latent[torch.arange(latent.size(0)), selected_indices] = 0
                    ablated_recon = model.decoder(abl_latent)

                    # Compare the outputs
                    pure_logits = val_data.intervene(seq, recon)
                    abl_logits = val_data.intervene(seq, ablated_recon)
                    causal_loss = -criterion_caus(F.log_softmax(abl_logits, dim=-1), F.softmax(pure_logits, dim=-1))

                    val_loss += (recon_loss.item() + causal_loss.item() * args.beta)
                    val_it += 1
                model.train()
                wandb.log({'recon_loss': recon_loss.item(),
                           'reg_loss'  : reg_loss.item() if args.alpha else 0,
                           'train_loss': train_loss,
                           'causal_loss': causal_loss.item(),
                           'val_loss'  : val_loss   / args.val_iters})

                if args.val_patience and val_loss > prev_loss:
                    loss_increasing += 1
                    if loss_increasing == args.val_patience: break
                else:
                    loss_increasing = 0
                    prev_loss = val_loss

                wandb.log({'recon_loss': recon_loss.item(),
                           'reg_loss'  : reg_loss.item() if args.alpha else 0,
                           'causal_loss': causal_loss.item(),
                           'train_loss': train_loss})
            train_it += 1

    i = 0
    while True:
        dir_name = os.path.join(args.model_dir, f'sae_{i}')
        if not os.path.exists(dir_name): break
        i += 1
    os.mkdir(dir_name)
    torch.save(model.state_dict(), os.path.join(dir_name, 'model.pth'))
    with open(os.path.join(dir_name, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    wandb.finish()
