import os, pandas as pd
P = "/mnt/f415dcc8-27a5-4e66-8493-25f1d939974c/internshipwork/journal/processed"
df   = pd.read_csv(f"{P}/banglaabusememe.csv").fillna("")
easy = pd.read_csv(f"{P}/bam_ocr_cache.csv").fillna("").set_index("image_path")["ocr_text"]

empty = df["text"].str.strip() == ""
df.loc[empty, "text"] = df.loc[empty, "image_path"].map(easy).fillna("")
df["ocr_source"] = "surya"
df.loc[empty & (df["text"].str.strip() != ""), "ocr_source"] = "easyocr"
df.to_csv(f"{P}/banglaabusememe.csv", index=False, encoding="utf-8")
print("Still empty:", (df["text"].str.strip() == "").sum())
print(df["ocr_source"].value_counts())