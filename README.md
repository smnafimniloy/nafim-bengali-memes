# NAFIM: Noise-Aware Fusion for Image–text Memes

**Robust multimodal detection of harmful Bengali memes under real-world OCR noise.**

> ⚠️ **Content warning:** this project studies hateful, abusive and aggressive memes. The datasets contain offensive content.

---

## Overview

Bengali harmful-meme detectors are usually trained and evaluated on **human-typed captions**. A real moderation system only has the meme image, so the text must be extracted with **OCR**, which is error-prone for Bengali memes (stylised fonts, conjunct characters, watermarks).

This repository:

1. builds a unified benchmark from three public Bengali meme datasets (**BHM**, **MIMOSA**, **BanglaAbuseMeme**);
2. measures OCR quality (CER / WER) of **Surya** and **EasyOCR** on Bengali memes against gold captions;
3. quantifies how much OCR noise degrades multimodal (image + text) detection;
4. proposes **NAFIM**, a fusion model that estimates how reliable the OCR text is and relies more on the image when the text is unreliable;
5. compares NAFIM with two published Bengali meme models (**DORA**, **MAF**) and a vision–language model (**Qwen2-VL-2B**).

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
├── 04_nafim.ipynb              # NAFIM vs. fusion baselines + paired bootstrap tests
├── 05_ablation_crossdataset.ipynb  # NAFIM ablations and cross-dataset transfer
├── 06_vlm_baseline.ipynb       # Qwen2-VL-2B zero-shot and QLoRA fine-tuning
├── 07_xai_error_analysis.ipynb # reliability analysis, modality reliance, Grad-CAM, shortcut check
├── 08_sota_comparison.ipynb    # DORA and MAF trained on gold and mixed text (3 seeds)
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

# extra packages for notebooks 06 (QLoRA) and 08 (DORA / XGLM tokenizer)
pip install "peft==0.13.2" "bitsandbytes==0.44.1" "accelerate==1.0.1" protobuf
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

Please respect each dataset's license and terms of use. This repository does not redistribute any images, captions or OCR text; the OCR scripts regenerate the OCR text from the original images.

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
# then run notebooks 01 -> 02 -> ... -> 08 (kernel: "Python (internship1)")
```

Long notebooks can be run headless and are resumable:

```bash
nohup jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=internship1 \
  03_ocr_eval.ipynb > nb03.log 2>&1 &
```

Experiments report mean ± std macro-F1 over 5 seeds (42–46); DORA and MAF use 3 seeds. Each results folder contains a `config.json` with the exact settings and library versions. Notebook 06 (QLoRA) takes about 3 hours per seed on an RTX 3050.

---

## Results

Macro-F1 (mean over seeds). *Mean* and *Worst* are taken over gold, Surya and EasyOCR text.

| Task | Model | Gold | Mean | Worst |
|---|---|---|---|---|
| BHM | Fusion-mixed | 0.723 | 0.721 | 0.716 |
| | **NAFIM** | **0.728** | **0.723** | **0.717** |
| MIMOSA (5-class) | Fusion-mixed | 0.777 | 0.770 | 0.761 |
| | **NAFIM** | **0.784** | **0.776** | **0.767** |
| MIMOSA (binary) | Fusion-mixed | 0.798 | 0.794 | 0.788 |
| | **NAFIM** | **0.815** | **0.808** | **0.798** |

- **OCR quality:** Surya CER is 29.8% on BHM and 38.6% on MIMOSA; Surya outperforms EasyOCR.
- **Effect on detection:** models trained on gold captions lose up to 4.6 macro-F1 (text-only) and 3.6 (multimodal) on OCR text, mostly on memes with CER ≥ 0.5.
- **Published models:** gold-trained DORA and MAF lose 5.5 and 5.9 points on MIMOSA 5-class with Surya text (8.0 and 6.4 with EasyOCR). NAFIM outperforms their mixed-text variants by 3.9–6.7 points in mean macro-F1 (paired bootstrap, p ≤ 0.006).
- **Reliability estimator:** the predicted reliability correlates with true OCR quality (Spearman ρ = 0.63–0.81), and NAFIM's reliance on text drops by about a third on poorly recognized memes.

Full tables, significance tests and figures are in `results/`.

---

## Citation

Citation will be added after publication.

## Acknowledgements

We thank the authors of BHM, MIMOSA and BanglaAbuseMeme for releasing their datasets, and the developers of Surya OCR, EasyOCR, CLIP, XLM-R, BanglaBERT and Qwen2-VL, and the authors of DORA and MAF for releasing their code.

## License

See [LICENSE](LICENSE). The license applies to the code in this repository only, not to the datasets.
