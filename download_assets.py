#!/usr/bin/env python3
"""
Download large model / dataset assets for the IBM Generative AI Engineering with LLMs Specialization.
These files are intentionally NOT stored in GitHub (size). Run after clone.

Usage:
  python download_assets.py              # download all missing
  python download_assets.py --force      # re-download all
  python download_assets.py --list       # list only
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (url, relative path under repo root, short label)
ASSETS = [
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/tQdezXocAJMBMPfUJx_iUg/glove-6B.zip',
     'Glove/glove-6B.zip',
     'GloVe embeddings (shared)'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0201EN-Coursera/RNN-TR-model.pt',
     'C02 - Gen AI Foundational Models for NLP & Language Understanding/Module 2/Labs/assets/lab3_RNN-TR-model.pt',
     'C02 RNN-TR model'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/kyn1_OsXrzjef0xihlsXmg.pt',
     'C03 - Generative AI Language Modeling with Transformers/Module 2/Labs/assets/lab1_decoder.pt',
     'C03 decoder'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/bZaoQD52DcMpE7-kxwAG8A.zip',
     'C03 - Generative AI Language Modeling with Transformers/Module 2/Labs/assets/lab2_bert_dataset.zip',
     'C03 BERT dataset'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/H04Cs7O75aOfmJ4YP2HdPw.pt',
     'C03 - Generative AI Language Modeling with Transformers/Module 2/Labs/assets/lab2_model.pt',
     'C03 BERT model'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0201EN-Coursera/transformer.pt',
     'C03 - Generative AI Language Modeling with Transformers/Module 2/Labs/assets/lab4_transformer.pt',
     'C03 transformer'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-AI0201EN-Coursera/transformer_de_to_en_model.pt',
     'C03 - Generative AI Language Modeling with Transformers/Module 2/Labs/assets/lab4_transformer_de_to_en_model.pt',
     'C03 DE→EN model'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/BeXRxFT2EyQAmBHvxVaMYQ/bert-scratch-model.pt',
     'C04 - Generative AI Engineering and Fine-Tuning Transformers/Module 1/Labs/assets/lab2_bert-scratch-model.pt',
     'C04 BERT scratch'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/Assistant_model.pt',
     'C04 - Generative AI Engineering and Fine-Tuning Transformers/Module 1/Labs/assets/lab4_Assistant_model.pt',
     'C04 Assistant model'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/wFhKpkBMSgjmZKRSyayvsQ/bert-classification-model.pt',
     'C04 - Generative AI Engineering and Fine-Tuning Transformers/Module 1/Labs/assets/lab4_bert-classification-model.pt',
     'C04 BERT classification'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/35t-FeC-2uN1ozOwPs7wFg.gz',
     'C04 - Generative AI Engineering and Fine-Tuning Transformers/Module 2/Labs/assets/lab1_dataset.gz',
     'C04 dataset'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/PGhd5G_NVrWNH-_jdjwNlw/model-adapters.pt',
     'C04 - Generative AI Engineering and Fine-Tuning Transformers/Module 2/Labs/assets/lab1_model-adapters.pth',
     'C04 adapters'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/9c3Dh2O_jsYBShBuchUNlg/model-AG%20News%20small1.pth',
     'C04 - Generative AI Engineering and Fine-Tuning Transformers/Module 2/Labs/assets/lab1_model-AG News small1.pth',
     'C04 AG News'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/e0WOHKh5dnrbC2lGhpsMMw/model-fine1.pth',
     'C04 - Generative AI Engineering and Fine-Tuning Transformers/Module 2/Labs/assets/lab1_model-fine1.pth',
     'C04 fine1'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/B-1H6lpDg-A0zRwpB6Ek2g/model-fine2.pth',
     'C04 - Generative AI Engineering and Fine-Tuning Transformers/Module 2/Labs/assets/lab1_model-fine2.pth',
     'C04 fine2'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/q66IH6a7lglkZ4haM6hB1w/model-IMDB%20dataset%20small2.pth',
     'C04 - Generative AI Engineering and Fine-Tuning Transformers/Module 2/Labs/assets/lab1_model-IMDB dataset small2.pth',
     'C04 IMDB'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZvhVWJU0flC7BmU1jjYxjg/model-imdb-freeze-true2.pth',
     'C04 - Generative AI Engineering and Fine-Tuning Transformers/Module 2/Labs/assets/lab2_model-imdb-freeze-true2.pth',
     'C04 IMDB freeze'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/JWPRb1RMhKLRMUWOKw9pxA/model-lora-final2.pth',
     'C04 - Generative AI Engineering and Fine-Tuning Transformers/Module 2/Labs/assets/lab2_model-lora-final2.pth',
     'C04 LoRA final'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/uGC04Pom651hQs1XrZ0NsQ/my-model-freeze-false.pth',
     'C04 - Generative AI Engineering and Fine-Tuning Transformers/Module 2/Labs/assets/lab2_my-model-freeze-false.pth',
     'C04 freeze false'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/8zCp__SHRSgGVlf5yP50Ag/ppo-bad-tar.gz',
     'C05 - Generative AI Advanced Fine-Tuning for LLMs/Module 2/Labs/assets/lab1_ppo-bad-tar.gz',
     'C05 PPO bad'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/we8t5N-45dVq3VhxGwYRAg/ppo-good-tar.gz',
     'C05 - Generative AI Advanced Fine-Tuning for LLMs/Module 2/Labs/assets/lab1_ppo-good-tar.gz',
     'C05 PPO good'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/gSWo8GeztngSmzHpqX_RaQ/ppo-good.pkl',
     'C05 - Generative AI Advanced Fine-Tuning for LLMs/Module 2/Labs/assets/lab1_ppo-good.pkl',
     'C05 PPO good pkl'),
    ('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/jMW99Z9mvxesgYR-H6y6Yw/ppo-bad.pkl',
     'C05 - Generative AI Advanced Fine-Tuning for LLMs/Module 2/Labs/assets/lab1_ppo-bad.pkl',
     'C05 PPO bad pkl'),
]


def download_one(url: str, dest: Path, label: str, force: bool = False) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"  [skip] {label}  ({dest.relative_to(ROOT)})")
        return True
    print(f"  [get ] {label}")
    print(f"         → {dest.relative_to(ROOT)}")
    try:
        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  [ok  ] {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        if dest.exists():
            dest.unlink()
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description="Download large course assets (models, datasets).")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists")
    parser.add_argument("--list", action="store_true", help="List assets only")
    args = parser.parse_args()

    print("=" * 70)
    print("  IBM LLMs Specialization — Asset Downloader")
    print("=" * 70)
    print(f"  Repo root: {ROOT}\n")

    if args.list:
        for url, rel, label in ASSETS:
            print(f"  • {label}\n    path: {rel}\n    url:  {url}\n")
        return 0

    ok = fail = 0
    for url, rel, label in ASSETS:
        dest = ROOT / rel
        if download_one(url, dest, label, force=args.force):
            ok += 1
        else:
            fail += 1
        print()

    print("=" * 70)
    print(f"  Done: {ok} ok, {fail} failed")
    print("=" * 70)
    if fail:
        print("  Tip: check network / URL availability and re-run.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
