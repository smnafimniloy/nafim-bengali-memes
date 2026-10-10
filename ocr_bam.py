import os
import pandas as pd
import easyocr
from tqdm import tqdm

ROOT = "/mnt/f415dcc8-27a5-4e66-8493-25f1d939974c/internshipwork/journal"
CSV = os.path.join(ROOT, "processed", "banglaabusememe.csv")
CACHE = os.path.join(ROOT, "processed", "bam_ocr_cache.csv")

df = pd.read_csv(CSV)
reader = easyocr.Reader(["bn", "en"], gpu=True)

# resume from cache if the run was interrupted
done = pd.read_csv(CACHE) if os.path.exists(CACHE) else pd.DataFrame(columns=["image_path", "ocr_text"])
done_set = set(done["image_path"])
results = done.to_dict("records")

todo = [p for p in df["image_path"] if p not in done_set]
for i, path in enumerate(tqdm(todo, desc="OCR")):
    try:
        text = " ".join(reader.readtext(path, detail=0, paragraph=True))
    except Exception as e:
        print("Error:", os.path.basename(path), e)
        text = ""
    results.append({"image_path": path, "ocr_text": text.strip()})
    if (i + 1) % 200 == 0:
        pd.DataFrame(results).to_csv(CACHE, index=False, encoding="utf-8")

pd.DataFrame(results).to_csv(CACHE, index=False, encoding="utf-8")

# merge OCR text into the dataset CSV
ocr = pd.read_csv(CACHE).fillna("")
df = df.drop(columns=["text"]).merge(ocr.rename(columns={"ocr_text": "text"}), on="image_path", how="left")
df["text"] = df["text"].fillna("")
df = df[["image_path", "text", "label", "label_bin", "label_name", "split",
         "sentiment", "sarcasm", "vulgar", "target"]]
df.to_csv(CSV, index=False, encoding="utf-8")

print("Empty text after OCR:", (df["text"].str.strip() == "").sum(), "/", len(df))
print(df[["text", "label_name"]].sample(5, random_state=42).to_string())