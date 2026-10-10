"""
Run Surya OCR and EasyOCR on BHM and MIMOSA images (datasets that have gold captions),
so OCR text can be compared against human-typed text.

Adds 3 columns to processed/bhm.csv and processed/mimosa.csv:
  text_surya    - Surya OCR (conf >= 0.5, cleaned)      [same settings as ocr_surya.py]
  text_easyocr  - EasyOCR bn+en, raw                    [same settings as ocr_bam.py]
  text_ocr      - Surya, with EasyOCR fallback when Surya is empty  [same rule as BanglaAbuseMeme]
The gold caption stays in the `text` column.

Run in the `ocr` env:  python ocr_gold_datasets.py
Resumable: progress is cached per dataset/engine; rerun to continue.
NOTE: if you rerun prepare_data.py later, these columns are lost -> rerun this script
      (it finishes instantly from the cache).
"""
import os, re, gc

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("RECOGNITION_BATCH_SIZE", "32")   # lower to 16 if CUDA out of memory
os.environ.setdefault("DETECTOR_BATCH_SIZE", "8")

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

ROOT = "/mnt/f415dcc8-27a5-4e66-8493-25f1d939974c/internshipwork/journal"
P = os.path.join(ROOT, "processed")
CACHE_DIR = os.path.join(P, "ocr_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

DATASETS = ["bhm", "mimosa"]
ENGINES = ["surya", "easyocr"]
SURYA_CONF = 0.5
SURYA_BATCH = 16          # lower to 8 if CUDA out of memory
SAVE_EVERY = 200
FAILED = {}               # images that could not be processed in this run


# ---------------- helpers ----------------
def cache_path(ds, engine):
    return os.path.join(CACHE_DIR, f"{ds}_{engine}.csv")


def load_cache(ds, engine):
    """Return successfully processed rows only. Rows from old caches (no `status` column)
    with empty text are dropped so they get re-processed once - they may be failed batches."""
    f = cache_path(ds, engine)
    if not os.path.exists(f):
        return [], set()
    c = pd.read_csv(f).fillna("").drop_duplicates("image_path", keep="last")
    if "status" not in c.columns:
        c["status"] = ["ok" if t.strip() else "retry" for t in c["ocr_text"].astype(str)]
    n_retry = int((c["status"] != "ok").sum())
    if n_retry:
        print(f"[{engine}] {ds}: {n_retry} cached rows were empty/failed -> will be re-processed")
    c = c[c["status"] == "ok"]
    return c.to_dict("records"), set(c["image_path"])


def save_cache(ds, engine, results):
    pd.DataFrame(results).to_csv(cache_path(ds, engine), index=False, encoding="utf-8")


def clean(text):
    text = re.sub(r"<[^>]+>", " ", text)                                         # html tags
    text = re.sub(r"(https?://\S+|www\.\S+|\S+\.(com|net|org)\S*)", " ", text)   # URLs
    text = re.sub(r"@\w+", " ", text)                                            # handles
    text = re.sub(r"[^ঀ-৿a-zA-Z0-9\s।,.!?'\"-]", " ", text)            # keep bn/en
    tokens = [t for t in text.split() if len(t) > 1 or re.match(r"[ঀ-৿]", t)]
    return " ".join(tokens).strip()


def load_img(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) < 1000:
        s = 1000 / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)
    elif max(w, h) > 2048:
        s = 2048 / max(w, h)
        img = img.resize((int(w * s), int(h * s)), Image.BICUBIC)
    return img


# ---------------- Surya ----------------
def run_surya(paths_by_ds):
    from surya.foundation import FoundationPredictor
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor

    foundation = FoundationPredictor()
    rec = RecognitionPredictor(foundation)
    det = DetectionPredictor()

    for ds, paths in paths_by_ds.items():
        failed = []
        results, done = load_cache(ds, "surya")
        todo = [p for p in paths if p not in done]
        print(f"\n[surya] {ds}: {len(done)} cached, {len(todo)} to do")
        for i in tqdm(range(0, len(todo), SURYA_BATCH), desc=f"surya {ds}"):
            batch = todo[i:i + SURYA_BATCH]
            try:
                preds = rec([load_img(p) for p in batch], det_predictor=det)
                for p, pred in zip(batch, preds):
                    lines = [l.text for l in pred.text_lines if (l.confidence or 0) >= SURYA_CONF]
                    results.append({"image_path": p, "ocr_text": clean(" ".join(lines)), "status": "ok"})
            except Exception as e:
                # batch failed (usually out of memory): free memory and retry images one by one
                print("batch error -> retrying one by one:", str(e)[:120])
                gc.collect(); torch.cuda.empty_cache()
                for p in batch:
                    try:
                        pred = rec([load_img(p)], det_predictor=det)[0]
                        lines = [l.text for l in pred.text_lines if (l.confidence or 0) >= SURYA_CONF]
                        results.append({"image_path": p, "ocr_text": clean(" ".join(lines)), "status": "ok"})
                    except Exception as e2:
                        failed.append(p)          # NOT cached -> retried on the next run
                        print("  failed:", os.path.basename(p), str(e2)[:80])
                        gc.collect(); torch.cuda.empty_cache()
            save_cache(ds, "surya", results)
        if failed:
            print(f"[surya] {ds}: {len(failed)} images failed - rerun the script to retry them")
            FAILED[f"surya/{ds}"] = len(failed)

    del foundation, rec, det
    gc.collect(); torch.cuda.empty_cache()


# ---------------- EasyOCR ----------------
def run_easyocr(paths_by_ds):
    import easyocr
    reader = easyocr.Reader(["bn", "en"], gpu=torch.cuda.is_available())

    for ds, paths in paths_by_ds.items():
        results, done = load_cache(ds, "easyocr")
        todo = [p for p in paths if p not in done]
        print(f"\n[easyocr] {ds}: {len(done)} cached, {len(todo)} to do")
        for i, p in enumerate(tqdm(todo, desc=f"easyocr {ds}")):
            try:
                text = " ".join(reader.readtext(p, detail=0, paragraph=True)).strip()
            except Exception as e:
                print("error:", os.path.basename(p), str(e)[:80])   # NOT cached -> retried next run
                FAILED[f"easyocr/{ds}"] = FAILED.get(f"easyocr/{ds}", 0) + 1
                gc.collect(); torch.cuda.empty_cache()
                continue
            results.append({"image_path": p, "ocr_text": text, "status": "ok"})
            if (i + 1) % SAVE_EVERY == 0:
                save_cache(ds, "easyocr", results)
        save_cache(ds, "easyocr", results)

    del reader
    gc.collect(); torch.cuda.empty_cache()


# ---------------- main ----------------
if __name__ == "__main__":
    dfs = {ds: pd.read_csv(os.path.join(P, f"{ds}.csv")).fillna("") for ds in DATASETS}
    paths_by_ds = {ds: df["image_path"].tolist() for ds, df in dfs.items()}

    if "surya" in ENGINES:
        run_surya(paths_by_ds)
    if "easyocr" in ENGINES:
        run_easyocr(paths_by_ds)

    if FAILED:
        print("\nSome images failed:", FAILED)
        print("Rerun this script (it only retries the missing ones). Not merging until all are done.")
        raise SystemExit(1)

    # merge into the processed CSVs
    for ds, df in dfs.items():
        df = df.drop(columns=[c for c in ["text_surya", "text_easyocr", "text_ocr"] if c in df.columns])
        for engine in ENGINES:
            c = pd.read_csv(cache_path(ds, engine)).fillna("").drop_duplicates("image_path", keep="last")
            c = c[["image_path", "ocr_text"]]
            missing = len(set(df["image_path"]) - set(c["image_path"]))
            assert missing == 0, f"{ds}/{engine}: {missing} images not processed yet - rerun the script"
            df = df.merge(c.rename(columns={"ocr_text": f"text_{engine}"}), on="image_path", how="left")
            df[f"text_{engine}"] = df[f"text_{engine}"].fillna("").astype(str)

        surya_empty = df["text_surya"].str.strip() == ""
        df["text_ocr"] = df["text_surya"]
        df.loc[surya_empty, "text_ocr"] = df.loc[surya_empty, "text_easyocr"]
        df.to_csv(os.path.join(P, f"{ds}.csv"), index=False, encoding="utf-8")

        print(f"\n{ds}: rows {len(df)} | empty surya {surya_empty.sum()} "
              f"| empty easyocr {(df['text_easyocr'].str.strip() == '').sum()} "
              f"| empty text_ocr {(df['text_ocr'].str.strip() == '').sum()}")
        print(df[["text", "text_ocr"]].sample(3, random_state=1).to_string())

    print("\nDone. Gold text is in `text`; OCR text in `text_surya`, `text_easyocr`, `text_ocr`.")