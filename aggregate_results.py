import os
import glob
import pandas as pd

LOG_DIR = "logs"
OUT_SUMMARY = os.path.join(LOG_DIR, "summary_val.csv")
OUT_TEX = os.path.join(LOG_DIR, "table_val.tex")


def pick_best_row(df: pd.DataFrame):
    required = ["miou", "macc", "fwiou"]
    for c in required:
        if c not in df.columns:
            return None

    # 若有多行，取 miou 最大行；若只有一行（如 official_ptv3），也兼容
    idx = df["miou"].astype(float).idxmax()
    return df.loc[idx]


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(LOG_DIR, "*.csv")))

    records = []
    for f in files:
        # 避免把自己输出再次读入造成循环污染
        if os.path.basename(f) in ["summary_val.csv"]:
            continue

        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[skip] read failed: {f} | {e}")
            continue

        if df.empty:
            print(f"[skip] empty file: {f}")
            continue

        best = pick_best_row(df)
        if best is None:
            print(f"[skip] missing required columns in: {f}")
            continue

        method = best["model"] if "model" in df.columns else os.path.splitext(os.path.basename(f))[0]

        records.append(
            {
                "Method": str(method),
                "mIoU": float(best["miou"]),
                "mAcc": float(best["macc"]),
                "fwIoU": float(best["fwiou"]),
                "log": f,
            }
        )

    if not records:
        print("No valid csv logs found in logs/*.csv")
        return

    out = pd.DataFrame(records)

    # 先按 mIoU 降序
    out = out.sort_values("mIoU", ascending=False).reset_index(drop=True)

    # 如果有 official_ptv3，放到第一行（作为基线展示更清晰）
    mask_official = out["Method"].astype(str).str.lower().eq("official_ptv3")
    if mask_official.any():
        official_row = out[mask_official]
        other_rows = out[~mask_official]
        out = pd.concat([official_row, other_rows], axis=0).reset_index(drop=True)

    print(out)
    out.to_csv(OUT_SUMMARY, index=False)

    latex = out.to_latex(index=False, float_format="%.4f")
    with open(OUT_TEX, "w") as f:
        f.write(latex)

    print(f"saved {OUT_SUMMARY} & {OUT_TEX}")


if __name__ == "__main__":
    main()