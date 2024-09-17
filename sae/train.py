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

            closest = torch.cdist(latent, latent, p=2).fill_diagonal_(float('inf')).argmin(-1)
            target_activn = activation[closest]
            target_logits = logits[closest]

            intervened_activn = args.step * (target_activn - activation) + activation
            intervened_logits = train_data.intervene(seq, intervened_activn)
            causal_loss = criterion(intervened_logits,
                                    args.step * (target_logits - logits) + logits)

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
                for activation, grad, seq in val_data:
                    if val_it > args.val_iters: break
                    activation = activation.to(device)
                    if 'input' in args.norm:
                        norm = torch.norm(activation, p=2, dim=-1)
                        activation = activation / norm.unsqueeze(-1)
                    latent, recon = model(activation)

                    recon_loss = criterion(recon, activation)

                    closest = torch.cdist(latent, latent, p=2).fill_diagonal_(float('inf')).argmin(-1)
                    target_activn = activation[closest]
                    target_logits = logits[closest]

                    intervened_activn = args.step * (target_activn - activation) + activation
                    intervened_logits = train_data.intervene(seq, intervened_activn)
                    causal_loss = criterion(intervened_logits,
                                            args.step * (target_logits - logits) + logits)

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
