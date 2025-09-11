import argparse, os, json, pandas as pd, numpy as np, torch
from utils import make_windows, rmse, mae, mape
import joblib
import matplotlib.pyplot as plt
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, horizon=30):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, horizon)
    def forward(self, x):
        out, _ = self.lstm(x); h_last = out[:, -1, :]
        return self.fc(h_last)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lookback", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=30)
    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.input, parse_dates=["date"])
    series = df["value"].values.astype("float32")
    scaler = joblib.load(os.path.join(args.outdir, "scaler.pkl"))
    from utils import scale_with, inverse_scale
    scaled = scale_with(series, scaler)
    X, y = make_windows(scaled, args.lookback, args.horizon)
    model = LSTMForecaster(horizon=args.horizon)
    state = torch.load(args.model, map_location="cpu")
    model.load_state_dict(state["model_state"] if isinstance(state, dict) and "model_state" in state else state)
    model.eval()
    last_input = torch.tensor(X[-1][:, None]).unsqueeze(0)
    with torch.no_grad():
        pred_scaled = model(last_input).numpy().flatten()
    pred = inverse_scale(pred_scaled, scaler)
    y_true = df["value"].values[-args.horizon:]
    r = {"rmse": rmse(y_true, pred), "mae": mae(y_true, pred), "mape": mape(y_true, pred)}
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(r, f, indent=2)
    dates = df["date"]
    pred_start = len(df) - args.horizon
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(dates, df["value"], label="actual")
    ax.plot(dates[pred_start:pred_start+args.horizon], pred, label="forecast")
    ax.set_title("Forecast vs Actual (Evaluate)"); ax.set_xlabel("Date"); ax.set_ylabel("Value")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(args.outdir, "forecast_plot.png"), dpi=160); plt.close(fig)
    print("[OK] Evaluation complete. Metrics saved.")

if __name__ == "__main__":
    main()
