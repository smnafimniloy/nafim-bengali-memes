# NAFIM: Noise-Aware Fusion for Image–text Memes

**Robust multimodal detection of harmful Bengali memes under real-world OCR noise.**

> ⚠️ **Content warning:** this project studies hateful, abusive and aggressive memes. The datasets contain offensive content.

> 🚧 **Status:** work in progress — paper in preparation.

---

## Overview

Bengali harmful-meme detectors are usually trained and evaluated on **human-typed captions**. A real moderation system only has the meme image, so the text must be extracted with **OCR**, which is error-prone for Bengali memes (stylised fonts, conjunct characters, watermarks).

This repository:

1. builds a unified benchmark from three public Bengali meme datasets (**BHM**, **MIMOSA**, **BanglaAbuseMeme**);
2. measures OCR quality (CER / WER) of **Surya** and **EasyOCR** on Bengali memes against gold captions;
3. quantifies how much OCR noise degrades multimodal (image + text) detection;
4. proposes **NAFIM**, a fusion model that estimates how reliable the OCR text is and relies more on the image when the text is unreliable.

### NAFIM in short

| Component | What it does |
|---|---|
| CER-supervised reliability estimator | predicts text reliability (1 − CER) from the text embedding + 11 text-quality features |
| Reliability-gated fusion | scales the text representation by the predicted reliability before fusing with the CLIP image feature |
| Text-source mixing | trains on gold, Surya, EasyOCR and synthetically corrupted text |
| Consistency loss | aligns predictions on noisy text with predictions on the gold text of the same meme |

Backbones: **XLM-R base** (text) + **CLIP ViT-L/14** (image, frozen features).

---

## Repository structure

```
.
├── prepare_data.py          # unify the 3 datasets -> processed/*.csv (+ BanglaAbuseMeme split, seed 42)
├── ocr_bam.py               # EasyOCR on BanglaAbuseMeme
├── ocr_surya.py             # Surya OCR on BanglaAbuseMeme
├── ocr_fallback.py          # Surya text with EasyOCR fallback for empty outputs
├── ocr_gold_datasets.py     # Surya + EasyOCR on BHM and MIMOSA (gold captions available)
├── 01_baselines_frozen.ipynb   # frozen CLIP / BanglaBERT features + logistic regression, cross-dataset
├── 02_finetuned.ipynb          # fine-tuned text / image / fusion baselines (5 seeds)
├── 03_ocr_eval.ipynb           # OCR quality (CER/WER) and gold-vs-OCR training/testing
├── 04_nafim.ipynb              # NAFIM vs. fusion baselines
└── results/                    # result tables (CSV), configs (JSON) and figures (PNG)
```

Not included in the repository: raw datasets, processed CSVs, extracted features and model predictions (see *Data*).

---

## Setup

Tested on Ubuntu with an NVIDIA RTX 3050 (8 GB). Two conda environments are used because Surya OCR needs a newer PyTorch.

**Main environment** (training, notebooks):

```bash
conda create -n internship1 python=3.10 -y
conda activate internship1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install "transformers==4.46.3" sentencepiece pandas openpyxl numpy scikit-learn scipy \
            pillow tqdm matplotlib rapidfuzz easyocr jupyter ipykernel
python -m ipykernel install --user --name internship1 --display-name "Python (internship1)"
```

**OCR environment** (Surya):

```bash
conda create -n ocr python=3.10 -y
conda activate ocr
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install surya-ocr==0.17.1 "transformers>=4.56.1,<4.57" requests pandas tqdm opencv-python-headless easyocr
```

---

## Data

Download the datasets from their original sources and place them under `dataset/`:

| Dataset | Task | Source |
|---|---|---|
| BHM — Bengali Hateful Memes (Hossain et al., ACL 2024) | hate vs. non-hate | https://github.com/eftekhar-hossain/Bengali-Hateful-Memes |
| MIMOSA (Ahsan et al., EACL 2024) | aggression type (5 classes) | https://github.com/shawlyahsan/Bengali-Aggression-Memes |
| BanglaAbuseMeme (Das & Mukherjee, EMNLP 2023) | abusive vs. non-abusive | https://github.com/hate-alert/BanglaAbuseMeme |

Expected layout:

```
dataset/
├── BHM/{Memes/, Files/}
├── BanglaAbuseMeme/{Images/, BanglaAbuseMeme_annotation.csv}
└── Bengali-Aggression-Memes-main/Dataset/{Img/, training_set.csv, validation_set.csv, testing_set.csv}
```

Please respect each dataset's license and terms of use. This repository does not redistribute any images or captions.

> Paths in the scripts and notebooks point to the author's machine — edit the `ROOT` variable at the top of each file.

---

## Reproducing the experiments

Run in this order:

```bash
conda activate internship1
python prepare_data.py

conda activate ocr
python ocr_surya.py && python ocr_fallback.py      # BanglaAbuseMeme text
python ocr_gold_datasets.py                        # OCR text for BHM and MIMOSA

conda activate internship1
# then run notebooks 01 -> 02 -> 03 -> 04 (kernel: "Python (internship1)")
```

Long notebooks can be run headless and are resumable:

```bash
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=internship1 \
  03_ocr_eval.ipynb > nb03.log 2>&1 &
```

All experiments use 5 seeds (42–46) and report mean ± std macro-F1; each results folder contains a `config.json` with the exact settings and library versions.

---

## Results

Final results will be added when the paper is complete. Preliminary findings:

- **OCR quality:** Surya CER is 29.8% on BHM and 38.6% on MIMOSA (median 18.7% / 25.8%); Surya outperforms EasyOCR.
- **Effect on detection:** models trained on gold captions lose up to 4.6 macro-F1 (text-only) and 3.6 (multimodal) when given OCR text; losses concentrate on memes with CER ≥ 0.5.

---

## Citation

Citation will be added after publication.

## Acknowledgements

We thank the authors of BHM, MIMOSA and BanglaAbuseMeme for releasing their datasets, and the developers of Surya OCR, EasyOCR, CLIP, XLM-R and BanglaBERT.

## License

See [LICENSE](LICENSE). The license applies to the code in this repository only, not to the datasets.
