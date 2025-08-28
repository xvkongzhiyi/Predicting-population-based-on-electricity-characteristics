# run.py (debug-enhanced)
import argparse, yaml, torch, sys, traceback, os, logging
from torch.utils.data import DataLoader
from powerts.preprocessing.pipeline import load_and_preprocess, train_valid_split
from powerts.data.dataset import SlidingWindowDataset
from powerts.models.dlinear import DLinear
from powerts.models.tsmixer import TSMixer
from powerts.models.timesfm_zero import TimesFMZeroShot
from powerts.trainer.train import train_loop
from powerts.utils.common import set_seed
from powerts.utils.metrics import mse, mae

# make transformers/chatty if present
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "info")
logging.basicConfig(level=logging.INFO)

def get_model(name, seq_len, pred_len, user_vocab, user_embeds=8, cfg=None):
    if name == 'dlinear':
        print("[get_model] instantiating DLinear")
        return DLinear(seq_len=seq_len, pred_len=pred_len, in_channels=2, user_embeds=user_embeds, user_vocab=user_vocab)
    if name == 'tsmixer':
        print("[get_model] instantiating TSMixer")
        return TSMixer(seq_len=seq_len, pred_len=pred_len, in_channels=2, user_embeds=user_embeds, user_vocab=user_vocab)
    if name == 'timesfm':
        print("[get_model] instantiating TimesFMZeroShot (zero-param fallback)")
        return TimesFMZeroShot(seq_len=seq_len, pred_len=pred_len)
    if name == 'time_llm':
        print("[get_model] instantiating TimeLLM (lazy import)")
        try:
            from powerts.models.time_llm import TimeLLM
        except Exception as e:
            print("[get_model] ERROR importing TimeLLM:", e)
            raise
        model_name = cfg.get('time_llm_name') if cfg else None
        return TimeLLM(model_name=(model_name or 'gpt2'), input_dim=2, seq_len=seq_len, pred_len=pred_len)
    raise ValueError('Unknown model: ' + name)

def evaluate_only(model, dataset, device='cpu', batch_size=64):
    print("[evaluate_only] start eval, dataset len:", len(dataset))
    if len(dataset) == 0:
        print("[evaluate_only] no samples, return")
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()
    mses, maes = [], []
    with torch.no_grad():
        for i, (x, u, y) in enumerate(loader):
            x = x.to(device); u = u.to(device); y = y.to(device)
            yhat = model(x, u)
            mses.append(mse(yhat.detach().cpu(), y.detach().cpu()))
            maes.append(mae(yhat.detach().cpu(), y.detach().cpu()))
            if i % 10 == 0:
                print(f"[evaluate_only] batch {i} done")
    print(f"[evaluate_only] mse={sum(mses)/len(mses):.6f} mae={sum(maes)/len(maes):.6f}")

def has_trainable_params(model):
    for p in model.parameters():
        if p is not None and p.numel() > 0 and p.requires_grad:
            return True
    return False

def main():
    try:
        print("[run.py] starting")
        ap = argparse.ArgumentParser()
        ap.add_argument('--config', type=str, default='configs/example.yaml')
        ap.add_argument('--data', type=str, default='data/power_sample.csv')
        ap.add_argument('--model', type=str, default='dlinear', help='[dlinear|tsmixer|timesfm|time_llm]')
        ap.add_argument('--seq_len', type=int, default=None)
        ap.add_argument('--pred_len', type=int, default=None)
        ap.add_argument('--epochs', type=int, default=None)
        args = ap.parse_args()
        print("[run.py] args:", args)

        if not os.path.exists(args.config):
            print(f"[run.py] ERROR: config file not found: {args.config}")
        cfg = yaml.safe_load(open(args.config)) if os.path.exists(args.config) else {}
        for k in ['model','seq_len','pred_len','epochs']:
            v = getattr(args, k, None)
            if v not in [None, 'None']:
                cfg[k] = v if v is not None else cfg.get(k)

        print("[run.py] loaded cfg:", cfg)
        set_seed(cfg.get('seed', 42))

        print("[run.py] loading and preprocessing data:", args.data)
        df = load_and_preprocess(args.data, norm=cfg.get('norm','standard'))
        train, valid = train_valid_split(df, valid_ratio=0.2)
        seq_len, pred_len = int(cfg['seq_len']), int(cfg['pred_len'])
        print(f"[run.py] seq_len={seq_len}, pred_len={pred_len}")

        train_ds = SlidingWindowDataset(train, seq_len, pred_len)
        valid_ds = SlidingWindowDataset(valid, seq_len, pred_len)
        user_vocab = len(train_ds.user2idx)
        print(f"[run.py] train samples: {len(train_ds)} valid samples: {len(valid_ds)} users: {user_vocab}")

        model = get_model(cfg['model'], seq_len, pred_len, user_vocab=user_vocab,
                          user_embeds=cfg.get('user_embed_dim',8), cfg=cfg)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print('Device:', device)
        # If model has no trainable params, skip training loop and only evaluate
        if not has_trainable_params(model):
            print("Model has no trainable parameters — skipping training and running evaluation-only.")
            evaluate_only(model, valid_ds, device=device, batch_size=cfg.get('batch_size', 64))
            return

        print("[run.py] entering training loop")
        train_loop(model, train_ds, valid_ds, epochs=int(cfg['epochs']), lr=float(cfg.get('lr', 1e-3)), device=device)
    except Exception as e:
        print("[run.py] EXCEPTION OCCURRED:", e)
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
