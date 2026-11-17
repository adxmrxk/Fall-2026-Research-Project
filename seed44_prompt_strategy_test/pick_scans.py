"""
Pick a random batch of scans from MSD Task07 Pancreas using a fixed seed.
Same seed + same dataset = same scans every time.

Usage:
    python pick_scans.py [seed] [count]
Writes sample_seed<seed>.txt (one scan name per line) in this folder.
"""
import os
import sys
import random

IMAGES = r"C:\Users\AdamR\Downloads\Task07_Pancreas\Task07_Pancreas\imagesTr"
OUT = os.path.dirname(os.path.abspath(__file__))

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 44
COUNT = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def main():
    names = sorted(
        n for n in os.listdir(IMAGES)
        if n.endswith(".nii.gz") and not n.startswith("._")
    )
    rng = random.Random(SEED)  # own generator, so other code can't change the picks
    chosen = sorted(rng.sample(names, COUNT))

    out_path = os.path.join(OUT, f"sample_seed{SEED}.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(n.replace(".nii.gz", "") for n in chosen) + "\n")

    print(f"Seed {SEED}: picked {len(chosen)} of {len(names)} scans")
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
