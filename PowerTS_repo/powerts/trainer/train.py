
import torch, time
from torch.utils.data import DataLoader
from ..utils.metrics import mse, mae

def train_loop(model, train_ds, valid_ds, epochs=3, lr=1e-3, device='cpu'):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=64, shuffle=False)
    best = None
    for ep in range(1, epochs+1):
        model.train()
        losses = []
        for x,u,y in train_loader:
            x,u,y = x.to(device), u.to(device), y.to(device)
            opt.zero_grad()
            yhat = model(x, u)
            loss = loss_fn(yhat, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # valid
        model.eval()
        with torch.no_grad():
            vs, va = [], []
            for x,u,y in valid_loader:
                x,u,y = x.to(device), u.to(device), y.to(device)
                yhat = model(x,u)
                vs.append(mse(yhat, y))
                va.append(mae(yhat, y))
        print(f"Epoch {ep}: train_loss={sum(losses)/len(losses):.4f} valid_mse={sum(vs)/len(vs):.4f} valid_mae={sum(va)/len(va):.4f}")
    return model
