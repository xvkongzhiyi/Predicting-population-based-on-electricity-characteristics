
import argparse, yaml, torch
from powerts.preprocessing.pipeline import load_and_preprocess, train_valid_split
from powerts.data.dataset import SlidingWindowDataset
from powerts.models.dlinear import DLinear
from powerts.models.tsmixer import TSMixer
from powerts.models.timesfm_zero import TimesFMZeroShot
from powerts.trainer.train import train_loop
from powerts.utils.common import set_seed

def get_model(name, seq_len, pred_len, user_vocab, user_embeds=8):
    if name == 'dlinear':
        return DLinear(seq_len=seq_len, pred_len=pred_len, in_channels=2, user_embeds=user_embeds, user_vocab=user_vocab)
    if name == 'tsmixer':
        return TSMixer(seq_len=seq_len, pred_len=pred_len, in_channels=2, user_embeds=user_embeds, user_vocab=user_vocab)
    if name == 'timesfm':
        return TimesFMZeroShot(seq_len=seq_len, pred_len=pred_len)
    raise ValueError('Unknown model: ' + name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='configs/example.yaml')
    ap.add_argument('--data', type=str, default='data/power_sample.csv')
    ap.add_argument('--model', type=str, default='dlinear')
    ap.add_argument('--seq_len', type=int, default=None)
    ap.add_argument('--pred_len', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for k in ['model','seq_len','pred_len','epochs']:
        v = getattr(args, k, None)
        if v not in [None, 'None']:
            cfg[k] = v if v is not None else cfg[k]

    set_seed(cfg.get('seed', 42))

    df = load_and_preprocess(args.data, norm=cfg.get('norm','standard'))
    train, valid = train_valid_split(df, valid_ratio=0.2)
    seq_len, pred_len = int(cfg['seq_len']), int(cfg['pred_len'])

    train_ds = SlidingWindowDataset(train, seq_len, pred_len)
    valid_ds = SlidingWindowDataset(valid, seq_len, pred_len)
    user_vocab = len(train_ds.user2idx)

    model = get_model(cfg['model'], seq_len, pred_len, user_vocab=user_vocab, user_embeds=cfg.get('user_embed_dim',8))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device, 'users:', user_vocab, 'train samples:', len(train_ds), 'valid samples:', len(valid_ds))
    train_loop(model, train_ds, valid_ds, epochs=int(cfg['epochs']), lr=cfg.get('lr',1e-3), device=device)

if __name__ == '__main__':
    main()
