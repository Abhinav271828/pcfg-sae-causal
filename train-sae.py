import argparse
from sae import *

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir",  help="Path to the model directory",
                    type=str,       default=None,             required=True)
parser.add_argument("--ckpt",       help="Checkpoint file",
                    type=str,       default='latest_ckpt.pt', required=False)
parser.add_argument("--layer_name", help="Layer name",
                    type=str,       default=None,             required=True)
parser.add_argument("--exp_factor", help="Expansion factor",
                    type=int,       default=1,                required=False)

sparsity = parser.add_mutually_exclusive_group(required=True)
sparsity.add_argument("--alpha",     help="L1 regularization coefficient",
                      type=float,    default=None,  required=False)
sparsity.add_argument("--k",         help="Sparsity level",
                      type=int,      default=None,  required=False)
sparsity.add_argument("--sparsemax", help="Use sparsemax instead of softmax",
                      type=str,      default=False, required=False)

parser.add_argument("--beta",        help="Causal loss coefficient",
                    type=float,      default=1,     required=False)
parser.add_argument("--causal_loss", help="Which causal loss to use",
                    type=str,        default='mse', required=False)
parser.add_argument("--pre_bias",    help="Use pre-bias",
                    type=eval,       default=False, required=False)
parser.add_argument("--norm",        help="Normalization method",
                    type=str,        default=None,  required=False)
parser.add_argument("--lr",          help="Learning rate",
                    type=float,      default=1e-6,  required=False)
parser.add_argument("--train_iters", help="Number of iterations to train",
                    type=int,        default=1000,  required=False)
parser.add_argument("--val_iters",   help="Number of iterations to validate",
                    type=int,        default=10,    required=False)
parser.add_argument("--val_interval",help="Validation interval",
                    type=int,        default=50,    required=False)

patience = parser.add_mutually_exclusive_group()
patience.add_argument("--patience",     help="Patience",
                      type=int,         default=None, required=False)
patience.add_argument("--val_patience", help="Validation patience",
                     type=int,          default=None, required=False)

parser.add_argument("--config", help="Name of the configuration file for DGP",
                    type=str,   default=False, required=False)

args = parser.parse_args()

train(args)