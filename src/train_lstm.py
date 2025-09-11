import argparse, os, json, pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils import make_windows, scale_series, rmse, mae, mape

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, horizon=30):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, horizon)
    def forward(self, x):
        out, _ = self.lstm(x)
        h_last = out[:, -1, :]
        y = self.fc(h_last)
        return y

def plot_curves(history, outpath):
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(history["train_loss"], label="train_loss")
    ax.plot(history["val_loss"], label="val_loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (MSE)"); ax.set_title("Training & Validation Loss")
    ax.legend(); fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)

def plot_forecast(dates, values, pred_start_idx, preds, outpath):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(dates, values, label="actual")
    fut_dates = dates[pred_start_idx:pred_start_idx+len(preds)]
    ax.plot(fut_dates, preds, label="forecast")
    ax.set_title("Forecast vs Actual"); ax.set_xlabel("Date"); ax.set_ylabel("Value")
    ax.legend(); fig.tight_layout(); fig.savefig(outpath, dpi=160); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--outdir", type=str, default="outputs")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input, parse_dates=["date"])
    series = df["value"].values.astype("float32")
    scaled, scaler = scale_series(series, os.path.join(args.outdir, "scaler.pkl"))
    X, y = make_windows(scaled, args.lookback, args.horizon)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    tr_ds = TensorDataset(torch.tensor(X_train[:, :, None]), torch.tensor(y_train))
    va_ds = TensorDataset(torch.tensor(X_val[:, :, None]), torch.tensor(y_val))
    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    model = LSTMForecaster(horizon=args.horizon)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.MSELoss()
    best_val = float("inf"); stale = 0; history = {"train_loss": [], "val_loss": []}
    best_path = os.path.join(args.outdir, "best_lstm.pt")
    for ep in range(1, args.epochs+1):
        model.train(); tloss = 0.0; n=0
        for xb, yb in tqdm(tr_dl, desc=f"Epoch {ep}/{args.epochs} [train]"):
            opt.zero_grad()
            preds = model(xb)
            loss = crit(preds, yb)
            loss.backward(); opt.step()
            tloss += loss.item() * xb.size(0); n += xb.size(0)
        tl = tloss / n
        model.eval(); vloss=0.0; vn=0
        with torch.no_grad():
            for xb, yb in tqdm(va_dl, desc=f"Epoch {ep}/{args.epochs} [val]"):
                preds = model(xb); loss = crit(preds, yb)
                vloss += loss.item() * xb.size(0); vn += xb.size(0)
        vl = vloss / vn
        history["train_loss"].append(tl); history["val_loss"].append(vl)
        print(f"[epoch {ep}] train_loss={tl:.4f} val_loss={vl:.4f}")
        if vl < best_val:
            best_val = vl; stale = 0
            torch.save({"model_state": model.state_dict(), "horizon": args.horizon, "lookback": args.lookback}, best_path)
        else:
            stale += 1
            if stale >= 5:
                print("Early stopping."); break
    plot_curves(history, os.path.join(args.outdir, "training_curves.png"))

    state = torch.load(best_path, map_location="cpu")
    model.load_state_dict(state["model_state"]); model.eval()
    last_input = torch.tensor(X[-1][:, None]).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(last_input).numpy().flatten()
    from utils import inverse_scale
    pred = inverse_scale(pred_scaled, scaler)
    pred_start = len(df) - args.horizon
    plot_forecast(df["date"], df["value"], pred_start, pred, os.path.join(args.outdir, "forecast_plot.png"))
    y_true = df["value"].values[-args.horizon:]
    r = {"rmse": rmse(y_true, pred), "mae": mae(y_true, pred), "mape": mape(y_true, pred)}
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(r, f, indent=2)
    print("[OK] Training complete. Metrics saved.")

if __name__ == "__main__":
    main()
