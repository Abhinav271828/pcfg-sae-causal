from sae import SAEData, SAE
import torch
import os
import pickle as pkl
from model import GPT
from dgp import get_dataloader
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
from evals import grammar_evals
import argparse

# Load model and data


parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", help="Path to the model directory",
                    type=str,      default=None,  required=True)
parser.add_argument("--start",     help="Start index of SAEs",
                    type=int,      default=None,  required=True)
parser.add_argument("--end",       help="End index of SAEs + 1",
                    type=int,      default=None,  required=True)
parser.add_argument("--iters",     help="Number of iterations",
                    type=int,      default=5)

args = parser.parse_args()
path = args.model_dir
start = args.start
end = args.end
iters = args.iters

state_dict = torch.load(os.path.join(path, 'latest_ckpt.pt'), map_location='cpu')
cfg = state_dict['config']

with open(os.path.join(path, 'grammar/PCFG.pkl'), 'rb') as f:
    pcfg = pkl.load(f)
model = GPT(cfg.model, pcfg.vocab_size)
model.load_state_dict(state_dict['net'])
model.eval()
dataloader = get_dataloader(
        language=cfg.data.language,
        config=cfg.data.config,
        alpha=cfg.data.alpha,
        prior_type=cfg.data.prior_type,
        num_iters=cfg.data.num_iters * cfg.data.batch_size,
        max_sample_length=cfg.data.max_sample_length,
        seed=cfg.seed,
        batch_size=cfg.data.batch_size,
        num_workers=0,
    )

# Load SAE
def get_config(idx):
    return json.load(open(os.path.join(path, F'sae_{idx}/config.json')))

def get_sae(idx):
    config = get_config(idx)
    data = SAEData(model_dir=path,
                   ckpt='latest_ckpt.pt',
                   layer_name=config['layer_name'],
                   config=config['config'],
                   device='cpu')
    embedding_size = data[0][0].size(-1)

    sae = SAE(embedding_size,
              config['exp_factor'] * embedding_size,
              pre_bias=config['pre_bias'],
              k=config['k'],
              sparsemax=config['sparsemax'],
              norm=config['norm'])
    state_dict = torch.load(os.path.join(path, f'sae_{idx}/model.pth'), map_location='cpu')
    sae.load_state_dict(state_dict)
    sae.eval()
    return sae

# Evaluate intervened accuracy
validities = []
stds = []
for i in tqdm(range(start, end)):
    print("Evaluating SAE", i)
    sae = get_sae(i)
    config = get_config(i)

    def hook(module, input, output):
        return sae(output)[1]

    match config['layer_name']:
        case 'wte':   module = model.transformer.wte
        case 'wpe':   module = model.transformer.wpe
        case 'attn0': module = model.transformer.h[0].attn
        case 'mlp0':  module = model.transformer.h[0].mlp
        case 'res0':  module = model.transformer.h[0]
        case 'attn1': module = model.transformer.h[1].attn
        case 'mlp1':  module = model.transformer.h[1].mlp
        case 'res1':  module = model.transformer.h[1]
        case 'ln_f':  module = model.transformer.ln_f
    handle = module.register_forward_hook(hook)

    current = []
    try:
      for _ in range(iters):
          results_after = grammar_evals(cfg, model, template=dataloader.dataset.template, grammar=dataloader.dataset.PCFG, device='cpu')
          current.append(results_after['validity'])
    except RuntimeError:
      with open(os.path.join(path, 'validities.txt'), 'a') as f:
          f.write(f"{i} nan\n")
      handle.remove()
      continue

    validity = torch.tensor(current).mean().item()
    std = torch.tensor(current).std().item()
    with open(os.path.join(path, 'validities.txt'), 'a') as f:
        f.write(f"{i} {validity} +- {std}\n")
    handle.remove()

