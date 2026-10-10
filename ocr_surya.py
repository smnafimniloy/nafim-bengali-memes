import os, re
os.environ.setdefault("RECOGNITION_BATCH_SIZE", "32")   # lower to 16 if you hit out-of-memory errors
os.environ.setdefault("DETECTOR_BATCH_SIZE", "8")

import pandas as pd
from PIL import Image
from tqdm import tqdm
from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

ROOT = "/mnt/f415dcc8-27a5-4e66-8493-25f1d939974c/internshipwork/journal"
CSV = os.path.join(ROOT, "processed", "banglaabusememe.csv")
CACHE = os.path.join(ROOT, "processed", "bam_ocr_surya.csv")
CONF = 0.5     # drop low-confidence lines
BATCH = 16

foundation = FoundationPredictor()
rec = RecognitionPredictor(foundation)
det = DetectionPredictor()

def load(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) < 1000:                                  # upscale small memes
        s = 1000 / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)
    elif max(w, h) > 2048:
        s = 2048 / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)
    return img

def clean(text):
    text = re.sub(r"<[^>]+>", " ", text)                                         # html tags
    text = re.sub(r"(https?://\S+|www\.\S+|\S+\.(com|net|org)\S*)", " ", text)   # URLs
    text = re.sub(r"@\w+", " ", text)                                            # handles
    text = re.sub(r"[^\u0980-\u09FFa-zA-Z0-9\s।,.!?'\"-]", " ", text)            # keep bn/en
    tokens = [t for t in text.split() if len(t) > 1 or re.match(r"[\u0980-\u09FF]", t)]
    return " ".join(tokens).strip()

df = pd.read_csv(CSV)
done = pd.read_csv(CACHE) if os.path.exists(CACHE) else pd.DataFrame(columns=["image_path", "ocr_text"])
done_set, results = set(done["image_path"]), done.to_dict("records")
todo = [p for p in df["image_path"] if p not in done_set]

for i in tqdm(range(0, len(todo), BATCH), desc="Surya OCR"):
    paths = todo[i:i + BATCH]
    try:
        preds = rec([load(p) for p in paths], det_predictor=det)
        for p, pred in zip(paths, preds):
            lines = [l.text for l in pred.text_lines if (l.confidence or 0) >= CONF]
            results.append({"image_path": p, "ocr_text": clean(" ".join(lines))})
    except Exception as e:
        print("Batch error:", e)
        for p in paths:
            results.append({"image_path": p, "ocr_text": ""})
    pd.DataFrame(results).to_csv(CACHE, index=False, encoding="utf-8")   # save progress

ocr = pd.read_csv(CACHE).fillna("")
df = df.drop(columns=["text"]).merge(ocr.rename(columns={"ocr_text": "text"}), on="image_path", how="left")
df["text"] = df["text"].fillna("")
df = df[["image_path", "text", "label", "label_bin", "label_name", "split",
         "sentiment", "sarcasm", "vulgar", "target"]]
df.to_csv(CSV, index=False, encoding="utf-8")

print("Empty text:", (df["text"].str.strip() == "").sum(), "/", len(df))
print(df[["text", "label_name"]].sample(5, random_state=1).to_string())