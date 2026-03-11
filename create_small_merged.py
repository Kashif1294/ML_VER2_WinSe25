"""
Create a small merged dataset (subset of rows and columns) for fast training.
Reads from merged_dataset.csv and writes a new CSV with fewer columns (every Nth) and optional row limit.
Samples rows across the file so all classes are represented.
"""
import os
import argparse
import random

def main():
    parser = argparse.ArgumentParser(description="Create small merged CSV for fast training")
    parser.add_argument("--input", default="Dataset/merged_dataset.csv", help="Merged CSV path")
    parser.add_argument("--output", default="Dataset/merged_small.csv", help="Output path")
    parser.add_argument("--max-rows", type=int, default=5000, help="Max rows to keep")
    parser.add_argument("--feature-step", type=int, default=50, help="Keep every Nth feature column (default 50 => ~500 features)")
    parser.add_argument("--n-cols", type=int, default=25017, help="Total columns in input CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling rows")
    args = parser.parse_args()

    label_col = 2
    # Keep columns 0, 1, 2 (so label stays at index 2) + every Nth feature
    feat_indices = [i for i in range(3, args.n_cols)][::args.feature_step]
    needed = [0, 1, 2] + feat_indices
    n_cols_out = len(needed)

    # First pass: count lines
    print("Counting lines...")
    with open(args.input, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)
    print(f"  Total lines: {total_lines}")

    if total_lines <= args.max_rows:
        keep_indices = set(range(total_lines))
    else:
        random.seed(args.seed)
        keep_indices = set(random.sample(range(total_lines), args.max_rows))

    print(f"Writing {len(keep_indices)} rows, {n_cols_out} columns (label + every {args.feature_step}th feature)")
    written = 0
    with open(args.input, "r", encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8", newline="") as fout:
        for idx, line in enumerate(fin):
            if idx not in keep_indices:
                continue
            parts = line.strip().split(",")
            if len(parts) < args.n_cols:
                continue
            out_parts = [parts[i] for i in needed]
            fout.write(",".join(out_parts) + "\n")
            written += 1
            if written % 1000 == 0:
                print(f"  {written} rows...")
    print(f"Done. Output: {args.output} ({written} rows, {n_cols_out} cols)")
    print(f"Train with: python train_all_models.py --data {args.output} --feature-step 1 --n-cols {n_cols_out} --max-samples {min(4000, written)}")

if __name__ == "__main__":
    main()
