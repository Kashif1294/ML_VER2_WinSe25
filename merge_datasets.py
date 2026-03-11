"""
Merge combined dataset files into one dataset for VER project.
Combines human_combined.csv, box_combined.csv, floor_combined.csv into a single CSV.
Uses chunked reading/writing to handle large files (multi-GB).
Labels: 0=floor, 1=box, 2=human (column index 2).
"""
import os
import argparse
from typing import Optional

def merge_combined_datasets(
    box_path: str,
    human_path: str,
    floor_path: str,
    output_path: str,
    chunk_size: int = 50000,
    max_rows_per_class: Optional[int] = None,
):
    """
    Merge the three class-specific CSVs into one CSV file.
    Each file is assumed to have label already in column index 2.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    files_and_labels = [
        (floor_path, 0, "floor"),
        (box_path, 1, "box"),
        (human_path, 2, "human"),
    ]

    total_written = 0
    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        for filepath, label, name in files_and_labels:
            if not os.path.isfile(filepath):
                print(f"  Skip (not found): {filepath}")
                continue
            print(f"\nReading {name} (label={label}) from {os.path.basename(filepath)}...")
            written = 0
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    while True:
                        lines = []
                        for _ in range(chunk_size):
                            line = f.readline()
                            if not line:
                                break
                            lines.append(line)
                        if not lines:
                            break
                        for line in lines:
                            parts = line.rstrip("\n").split(",")
                            if len(parts) > 2:
                                parts[2] = str(label)
                            out_f.write(",".join(parts) + "\n")
                            written += 1
                            total_written += 1
                        if max_rows_per_class and written >= max_rows_per_class:
                            break
                        if written % 100000 == 0 and written > 0:
                            print(f"    Written {written:,} rows ({name})...")
            except Exception as e:
                print(f"    Error reading {filepath}: {e}")
                raise
            print(f"  Done {name}: {written:,} rows (label={label})")
            if max_rows_per_class and written >= max_rows_per_class:
                print(f"    (capped at {max_rows_per_class:,} per class)")

    print(f"\nMerge complete. Total rows: {total_written:,}")
    print(f"  Output: {output_path}")
    return total_written


def main():
    parser = argparse.ArgumentParser(description="Merge VER combined datasets into one CSV")
    parser.add_argument("--box", default="Dataset/box_combined.csv", help="Path to box CSV")
    parser.add_argument("--human", default="Dataset/human_combined.csv", help="Path to human CSV")
    parser.add_argument("--floor", default="Dataset/floor_combined.csv", help="Path to floor CSV")
    parser.add_argument("-o", "--output", default="Dataset/merged_dataset.csv", help="Output merged CSV path")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Rows per chunk")
    parser.add_argument("--max-rows-per-class", type=int, default=None,
                        help="Max rows per class (default: all). Use for quick tests.")
    args = parser.parse_args()

    print("VER Project: Merge Combined Datasets")
    print("Label scheme: 0=floor, 1=box, 2=human (column index 2)")

    merge_combined_datasets(
        box_path=args.box,
        human_path=args.human,
        floor_path=args.floor,
        output_path=args.output,
        chunk_size=args.chunk_size,
        max_rows_per_class=args.max_rows_per_class,
    )


if __name__ == "__main__":
    main()
