import os
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = "/mnt/f415dcc8-27a5-4e66-8493-25f1d939974c/internshipwork/journal"
DATA = os.path.join(ROOT, "dataset")
OUT_DIR = os.path.join(ROOT, "processed")
os.makedirs(OUT_DIR, exist_ok=True)
SEED = 42
COLS = ["image_path", "text", "label", "label_bin", "label_name", "split"]


def finalize(df, name, extra_cols=()):
    df["text"] = df["text"].fillna("").astype(str).str.strip()
    exists = df["image_path"].apply(os.path.exists)
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    print("Total rows:", len(df), "| Missing images:", (~exists).sum(),
          "| Empty captions:", (df["text"] == "").sum(),
          "| Unmapped labels:", df["label"].isna().sum())
    if (~exists).any():
        print("  e.g. missing:", df.loc[~exists, "image_path"].head(5).map(os.path.basename).tolist())

    df = df[exists & df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    df["label_bin"] = df["label_bin"].astype(int)

    # duplicate / leakage check
    dup = df["image_path"].duplicated(keep=False)
    print("Duplicate image rows:", dup.sum())
    sets = {s: set(df.loc[df["split"] == s, "image_path"]) for s in ["train", "val", "test"]}
    print("Overlap train-val:", len(sets["train"] & sets["val"]),
          "| train-test:", len(sets["train"] & sets["test"]),
          "| val-test:", len(sets["val"] & sets["test"]))

    df = df[COLS + list(extra_cols)]
    out = os.path.join(OUT_DIR, f"{name}.csv")
    df.to_csv(out, index=False, encoding="utf-8")
    print("Saved:", out, "| rows:", len(df))
    print(pd.crosstab(df["split"], df["label_name"], margins=True))
    return df


# ---------------- 1. BHM ----------------
bhm_dir = os.path.join(DATA, "BHM")
bhm_img = os.path.join(bhm_dir, "Memes")
target_map = {"TI": "Individual", "TO": "Organization", "TC": "Community", "TS": "Society"}

rows = []
for split, tag in [("train", "train"), ("val", "valid"), ("test", "test")]:
    t1 = pd.read_excel(os.path.join(bhm_dir, "Files", f"{tag}_task1.xlsx"))
    t2 = pd.read_excel(os.path.join(bhm_dir, "Files", f"{tag}_task2.xlsx"))
    for t in (t1, t2):
        t["image_name"] = t["image_name"].astype(str).str.strip()
        t["Labels"] = t["Labels"].astype(str).str.strip()
    t2 = t2.drop_duplicates("image_name")[["image_name", "Labels"]].rename(columns={"Labels": "target"})
    t1 = t1.merge(t2, on="image_name", how="left")   # target only exists for hate memes
    t1["split"] = split
    rows.append(t1)

bhm = pd.concat(rows, ignore_index=True)
bhm["image_path"] = bhm["image_name"].apply(lambda x: os.path.join(bhm_img, x))
bhm = bhm.rename(columns={"Captions": "text", "Labels": "label_name"})
bhm["label"] = bhm["label_name"].map({"non-hate": 0, "hate": 1})
bhm["label_bin"] = bhm["label"]
bhm["target"] = bhm["target"].map(target_map).fillna("None")
finalize(bhm, "bhm", extra_cols=["target"])


# ---------------- 2. BanglaAbuseMeme ----------------
bam_dir = os.path.join(DATA, "BanglaAbuseMeme")
bam = pd.read_csv(os.path.join(bam_dir, "BanglaAbuseMeme_annotation.csv"))
bam["Ids"] = bam["Ids"].astype(str).str.strip()
bam["image_path"] = bam["Ids"].apply(lambda x: os.path.join(bam_dir, "Images", x))
bam["text"] = ""   # no captions provided -> fill later with OCR
bam["label_name"] = bam["abuse"].astype(str).str.strip()
bam["label"] = bam["label_name"].map({"Non-abusive": 0, "Abusive": 1})
bam["label_bin"] = bam["label"]
bam["target"] = bam["target"].fillna("None")

# stratified 70/10/20 split (no official split)
tr, te = train_test_split(bam.index, test_size=0.20, stratify=bam["label"], random_state=SEED)
tr, va = train_test_split(tr, test_size=0.125, stratify=bam.loc[tr, "label"], random_state=SEED)
bam["split"] = "train"
bam.loc[va, "split"] = "val"
bam.loc[te, "split"] = "test"
finalize(bam, "banglaabusememe", extra_cols=["sentiment", "sarcasm", "vulgar", "target"])


# ---------------- 3. MIMOSA ----------------
mim_dir = os.path.join(DATA, "Bengali-Aggression-Memes-main", "Dataset")
classes = ["non-aggressive", "political aggression", "religious aggression", "gendered aggression", "others"]

rows = []
for split, fname in [("train", "training_set.csv"), ("val", "validation_set.csv"), ("test", "testing_set.csv")]:
    d = pd.read_csv(os.path.join(mim_dir, fname))
    d["split"] = split
    rows.append(d)

mim = pd.concat(rows, ignore_index=True)
mim["image_name"] = mim["image_name"].astype(str).str.strip()
mim["image_path"] = mim["image_name"].apply(lambda x: os.path.join(mim_dir, "Img", x))
mim = mim.rename(columns={"Captions": "text", "Label": "label_name"})
mim["label_name"] = mim["label_name"].astype(str).str.strip()
mim["label"] = mim["label_name"].map({c: i for i, c in enumerate(classes)})   # 5-class
mim["label_bin"] = (mim["label_name"] != "non-aggressive").astype(int)
finalize(mim, "mimosa")

print("\nDone. Note: banglaabusememe.csv has empty text -> run OCR next.")
