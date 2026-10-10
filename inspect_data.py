import os, glob
import pandas as pd

ROOT = "/mnt/f415dcc8-27a5-4e66-8493-25f1d939974c/internshipwork/journal/dataset"
IMG_EXT = (".jpg", ".jpeg", ".png")

for ds in ["BHM", "BanglaAbuseMeme", "Bengali-Aggression-Memes-main/Dataset"]:
    path = os.path.join(ROOT, ds)
    print(f"\n{'='*60}\n{ds}\n{'='*60}")

    if not os.path.isdir(path):
        print("FOLDER NOT FOUND:", path)
        continue

    # folder structure (2 levels)
    for d, subdirs, files in os.walk(path):
        depth = d.replace(path, "").count(os.sep)
        if depth > 2 or "/.git" in d:
            continue
        imgs = sum(f.lower().endswith(IMG_EXT) for f in files)
        print(f"{'  '*depth}{os.path.basename(d) or ds}/  ({len(files)} files, {imgs} images)")

    # every CSV / Excel file
    for f in glob.glob(f"{path}/**/*.csv", recursive=True) + glob.glob(f"{path}/**/*.xlsx", recursive=True):
        try:
            df = pd.read_csv(f) if f.endswith(".csv") else pd.read_excel(f)
        except Exception as e:
            print(f"\n-- {f}: could not read ({e})")
            continue
        print(f"\n-- {os.path.relpath(f, path)}  shape={df.shape}")
        print("columns:", list(df.columns))
        print(df.head(2).to_string())
        for c in df.columns:
            if 1 < df[c].nunique() <= 10:
                print(f"  {c}: {df[c].value_counts().to_dict()}")
