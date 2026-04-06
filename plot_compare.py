import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

files = sorted(glob.glob("logs/*.csv"))
plotted = 0

for f in files:
    try:
        df = pd.read_csv(f)
    except Exception as e:
        print(f"[skip] read failed: {f} | {e}")
        continue

    # 只画包含 epoch + miou 的训练日志
    if ("epoch" not in df.columns) or ("miou" not in df.columns):
        print(f"[skip] no epoch/miou: {f}")
        continue

    if df.empty:
        print(f"[skip] empty: {f}")
        continue

    model_col = df["model"] if "model" in df.columns else None
    model_name = str(model_col.iloc[0]) if model_col is not None and len(model_col) > 0 else "unknown"
    stem = os.path.splitext(os.path.basename(f))[0]
    label = f"{model_name} | {stem}"

    plt.plot(df["epoch"], df["miou"], label=label, linewidth=1.8)
    plotted += 1

if plotted == 0:
    print("No valid training logs found (need columns: epoch, miou).")
else:
    plt.xlabel("Epoch")
    plt.ylabel("Val mIoU")
    plt.title("Model Comparison (Val mIoU)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = "logs/compare_miou.png"
    plt.savefig(out, dpi=200)
    print(f"[saved] {out}")

plt.show()