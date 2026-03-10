"""
eval_xm3600.py — Evaluate Task 1 checkpoint on XM3600

Evaluates the English-trained projection + LoRA checkpoint on XM3600
for English and all languages in the English typological cluster.

English cluster (k=8 clustering on URIEL syntactic features):
    en, de, fr, nl, da, sv, no, it, pt, ro, es

For English: standard generation, scored against English references.
For other languages: zero-shot forced generation via language prefix token,
    scored against native-language references.
    Expected to score low — this is the pre-multilingual-training baseline.
    The gap between English and other languages motivates Stage 2 (WiT training).

Dataset: floschne/xm3600 — each language is a separate split.
         3600 images per language, human-generated captions (not translated).

Usage:
    # Download on hastings first (has internet):
    python3 eval_xm3600.py --download_only

    # Evaluate on landonia (offline):
    python3 eval_xm3600.py --checkpoint outputs/best_checkpoint.pt

    # Evaluate specific languages only:
    python3 eval_xm3600.py --langs en de fr

    # Skip feature recomputation if already cached:
    python3 eval_xm3600.py  # uses xm3600_features.pt if present
"""

import argparse
import gc
import os
import random
import json
import torch
import torch.nn as nn
from transformers import (
    Blip2Processor,
    Blip2Model,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import load_dataset, Image as HFImage
from peft import get_peft_model, LoraConfig, TaskType
from tqdm import tqdm

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint",    default="outputs/best_checkpoint.pt")
parser.add_argument("--feature_file",  default="xm3600_features.pt")
parser.add_argument("--eval_batch",    type=int, default=16)
parser.add_argument("--n_qualitative", type=int, default=5)
parser.add_argument("--download_only", action="store_true",
                    help="Only download/cache the dataset, do not run eval. "
                         "Run this on hastings (has internet).")
parser.add_argument("--langs", nargs="+",
                    default=["en", "de", "fr", "nl", "da", "sv", "no", "it", "pt", "ro", "es"],
                    help="Languages to evaluate (must be in XM3600)")
args = parser.parse_args()

# English cluster — all present in XM3600
EN_CLUSTER = ["en", "de", "fr", "nl", "da", "sv", "no", "it", "pt", "ro", "es"]

# Language prefix tokens — used to force mT5 generation in non-English languages
# mT5 tokenises these and we use the first token as forced_bos_token_id
LANG_PREFIXES = {
    "en": "English:",
    "de": "Deutsch:",
    "fr": "Français:",
    "nl": "Nederlands:",
    "da": "Dansk:",
    "sv": "Svenska:",
    "no": "Norsk:",
    "it": "Italiano:",
    "pt": "Português:",
    "ro": "Română:",
    "es": "Español:",
}

# ── Download mode ─────────────────────────────────────────────────────────────
if args.download_only:
    print("Download mode — run this on hastings (login node with internet)")
    print("Use snapshot_download to cache XM3600:\n")
    print("  python3 -c \"")
    print("  import os")
    print("  from huggingface_hub import snapshot_download")
    print("  target = os.path.expanduser('~/.cache/huggingface/datasets/floschne___xm3600')")
    print("  snapshot_download(repo_id='floschne/xm3600', repo_type='dataset', local_dir=target)")
    print("  print('Done.')\"")
    print("\nThen verify with:")
    print("  ls ~/.cache/huggingface/datasets/floschne___xm3600/data/")
    exit(0)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory / 1e9
    print(f"GPU: {props.name} | VRAM: {total_vram:.1f} GB")

def vram_used():
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved(0) / 1e9
    return 0.0

# ── 1. Load XM3600 images (English split for image loading) ───────────────────
# XM3600 has the same 3600 images across all languages — we load images from
# the English split (most reliable) and use image_id to align across languages.
print("\n── 1. Load XM3600 Images " + "─"*45)
os.environ["HF_DATASETS_OFFLINE"] = "1"

print("  Loading English split for images...")
XM3600_DATA = os.path.expanduser("~/.cache/huggingface/datasets/floschne___xm3600/data")
assert os.path.isdir(XM3600_DATA), \
    f"XM3600 not found at {XM3600_DATA} — run snapshot_download on hastings first"

ds_en = load_dataset(
    "parquet",
    data_files={"en": f"{XM3600_DATA}/en-*.parquet"},
    split="en",
)
# images are stored as {'bytes': b'...'} — decode to PIL
ds_en = ds_en.cast_column("image", HFImage())

print(f"  {len(ds_en)} images | columns: {ds_en.column_names}")
print(f"  Sample image_id: {ds_en[0]['image_id']}")

# Build image_id → dataset index mapping for alignment across language splits
image_id_to_idx = {sample["image_id"]: i for i, sample in enumerate(ds_en)}
n_images = len(ds_en)

# ── 2. Load reference captions for all eval languages ─────────────────────────
print("\n── 2. Load Reference Captions " + "─"*40)
lang_refs = {}   # lang → {image_id: [caption, ...]}

for lang in args.langs:
    print(f"  Loading {lang} ...")
    ds_lang = load_dataset(
        "parquet",
        data_files={lang: f"{XM3600_DATA}/{lang}-*.parquet"},
        split=lang,
    )
    refs = {}
    for sample in ds_lang:
        iid = sample["image_id"]
        # captions field is a list of strings
        caps = sample["captions"]   # confirmed column name from dataset inspection
        refs[iid] = caps
    lang_refs[lang] = refs
    print(f"    {len(refs)} image_ids, {sum(len(v) for v in refs.values())} total captions")

# ── 3. Precompute Q-Former features ───────────────────────────────────────────
print("\n── 3. Precompute Q-Former Features " + "─"*34)

if os.path.exists(args.feature_file):
    print(f"  Found {args.feature_file} — loading from cache.")
    saved        = torch.load(args.feature_file, weights_only=False)
    all_features = saved["features"]
    saved_ids    = saved["image_ids"]
    # Rebuild image_id → feature index mapping
    feat_id_to_idx = {iid: i for i, iid in enumerate(saved_ids)}
    print(f"  Loaded {len(all_features)} feature vectors")
    assert len(all_features) == n_images, \
        f"Feature count {len(all_features)} != dataset size {n_images} — delete {args.feature_file} and rerun"

else:
    print("  Loading BLIP-2 for feature extraction...")
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    blip2_model = Blip2Model.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    for param in blip2_model.parameters():
        param.requires_grad = False
    blip2_model.eval()
    print(f"  BLIP-2 loaded | VRAM: {vram_used():.1f} GB")

    images       = [ds_en[i]["image"] for i in range(n_images)]
    image_ids    = [ds_en[i]["image_id"] for i in range(n_images)]
    all_feats    = []
    PRECOMPUTE_BS = 32

    for i in tqdm(range(0, n_images, PRECOMPUTE_BS), desc="Precomputing"):
        batch_imgs = images[i : i + PRECOMPUTE_BS]
        inputs = blip2_processor(images=batch_imgs, return_tensors="pt").to(device)

        with torch.no_grad():
            vision_out = blip2_model.vision_model(
                pixel_values=inputs.pixel_values.half(),
                return_dict=True,
            )
            image_embeds = vision_out.last_hidden_state
            image_attn   = torch.ones(
                image_embeds.shape[:-1], dtype=torch.long, device=device
            )
            query_tokens = blip2_model.query_tokens.expand(
                image_embeds.shape[0], -1, -1
            )
            qformer_out = blip2_model.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_attn,
                return_dict=True,
            )
            feats = qformer_out.last_hidden_state   # (B, 32, 768)

        all_feats.append(feats.cpu().float())

    all_features = torch.cat(all_feats, dim=0)   # (3600, 32, 768)
    feat_id_to_idx = {iid: i for i, iid in enumerate(image_ids)}
    print(f"  Features shape: {all_features.shape}")

    torch.save({
        "features":  all_features,
        "image_ids": image_ids,
    }, args.feature_file)
    print(f"  Saved: {args.feature_file}")

    del blip2_model, blip2_processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  BLIP-2 deleted | VRAM free: {total_vram - vram_used():.1f} GB")

assert not torch.isnan(all_features).any(), "NaNs in features"
assert not torch.isinf(all_features).any(), "Infs in features"
print("  Feature health check ✓")

# ── 4. Load mT5 + LoRA ────────────────────────────────────────────────────────
print("\n── 4. Load mT5-base + LoRA " + "─"*43)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
mt5_model     = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-base").to(device)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16, lora_alpha=32, lora_dropout=0.1,
    target_modules=["q", "v"],
)
mt5_model = get_peft_model(mt5_model, lora_config)

# ── 5. Projection MLP ─────────────────────────────────────────────────────────
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int = 768, out_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )
    def forward(self, x):
        return self.net(x)

projection = ProjectionMLP().to(device)

# ── 6. Load checkpoint ────────────────────────────────────────────────────────
print(f"\n── 6. Load Checkpoint " + "─"*47)
assert os.path.exists(args.checkpoint), \
    f"Checkpoint not found: {args.checkpoint}"

ckpt = torch.load(args.checkpoint, weights_only=True)
projection.load_state_dict(ckpt["projection"])
mt5_model.load_state_dict(ckpt["lora"])
projection.eval()
mt5_model.eval()
print(f"  {args.checkpoint} loaded ✓")

# ── 7. Build forced_bos_token_id for each language ───────────────────────────
print("\n── 7. Language prefix tokens " + "─"*40)
lang_bos = {}
for lang in args.langs:
    prefix     = LANG_PREFIXES[lang]
    prefix_ids = mt5_tokenizer(
        prefix, return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
    lang_bos[lang] = prefix_ids[0].item()
    print(f"  {lang}: '{prefix}' → forced_bos={lang_bos[lang]}")

# ── 8. Generate captions and score per language ───────────────────────────────
print("\n── 8. Evaluation " + "─"*52)
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu

# Only score images that have references in ALL eval languages
# (XM3600 is complete so this should be all 3600)
eval_image_ids = sorted(image_id_to_idx.keys())
print(f"  Evaluating on {len(eval_image_ids)} images × {len(args.langs)} languages\n")

results = {}   # lang → {cider, bleu1..4}

for lang in args.langs:
    print(f"── Language: {lang} " + "─"*50)
    refs_for_lang  = lang_refs[lang]
    forced_bos     = lang_bos[lang]
    is_english     = (lang == "en")

    preds_dict = {}
    refs_dict  = {}

    for start in tqdm(range(0, len(eval_image_ids), args.eval_batch),
                      desc=f"{lang}", leave=False):
        batch_ids  = eval_image_ids[start : start + args.eval_batch]
        # look up feature index for each image_id
        feat_idx   = [feat_id_to_idx[iid] for iid in batch_ids]
        features   = torch.stack([all_features[i] for i in feat_idx]).to(device)

        with torch.no_grad():
            projected = projection(features)

            if is_english:
                # No forced token for English — let model generate freely
                out = mt5_model.generate(
                    inputs_embeds=projected,
                    max_new_tokens=50,
                    num_beams=4,
                )
            else:
                # Force generation to start in target language
                out = mt5_model.generate(
                    inputs_embeds=projected,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=50,
                    num_beams=4,
                )

        for k, iid in enumerate(batch_ids):
            pred  = mt5_tokenizer.decode(out[k], skip_special_tokens=True)
            refs  = refs_for_lang.get(iid, [])
            if not refs:
                continue
            g              = start + k
            preds_dict[g]  = [pred]
            refs_dict[g]   = refs

    # Score
    cider_scorer   = Cider()
    cider_score, _ = cider_scorer.compute_score(refs_dict, preds_dict)
    bleu_scorer    = Bleu(4)
    bleu_scores, _ = bleu_scorer.compute_score(refs_dict, preds_dict)

    results[lang] = {
        "cider":  round(cider_score * 100, 2),
        "bleu1":  round(bleu_scores[0] * 100, 2),
        "bleu2":  round(bleu_scores[1] * 100, 2),
        "bleu3":  round(bleu_scores[2] * 100, 2),
        "bleu4":  round(bleu_scores[3] * 100, 2),
        "n":      len(preds_dict),
        "preds":  preds_dict,   # kept for qualitative display below
        "refs":   refs_dict,
    }

    print(f"  CIDEr: {results[lang]['cider']:.2f} | "
          f"BLEU-1: {results[lang]['bleu1']:.2f} | "
          f"BLEU-4: {results[lang]['bleu4']:.2f} | "
          f"n={results[lang]['n']}")

# ── 9. Summary table ──────────────────────────────────────────────────────────
print("\n" + "="*65)
print("SUMMARY — XM3600 English Cluster Evaluation")
print(f"Checkpoint: {args.checkpoint}")
print(f"Model trained on: English Flickr8k only")
print(f"Non-English: zero-shot forced generation via language prefix")
print("="*65)
print(f"{'Lang':<6} {'CIDEr':>7} {'BLEU-1':>7} {'BLEU-2':>7} {'BLEU-3':>7} {'BLEU-4':>7}  {'Note'}")
print("-"*65)
for lang in args.langs:
    r    = results[lang]
    note = "" if lang == "en" else "zero-shot"
    print(f"{lang:<6} {r['cider']:>7.2f} {r['bleu1']:>7.2f} {r['bleu2']:>7.2f} "
          f"{r['bleu3']:>7.2f} {r['bleu4']:>7.2f}  {note}")

# ── 10. Qualitative samples ───────────────────────────────────────────────────
print(f"\n── 10. Qualitative Samples (first {args.n_qualitative} images) " + "─"*20)
print("Same images shown across all languages for comparison.\n")

sample_ids = eval_image_ids[:args.n_qualitative]

# Regenerate captions directly for these images — avoids any index mapping issues
for iid in sample_ids:
    feat_i   = feat_id_to_idx[iid]
    feature  = all_features[feat_i].unsqueeze(0).to(device)

    print(f"Image: {iid}")
    with torch.no_grad():
        projected = projection(feature)
        for lang in args.langs:
            if lang == "en":
                out = mt5_model.generate(
                    inputs_embeds=projected,
                    max_new_tokens=50,
                    num_beams=4,
                )
            else:
                out = mt5_model.generate(
                    inputs_embeds=projected,
                    forced_bos_token_id=lang_bos[lang],
                    max_new_tokens=50,
                    num_beams=4,
                )
            pred = mt5_tokenizer.decode(out[0], skip_special_tokens=True)
            ref  = lang_refs[lang].get(iid, ["?"])[0]
            print(f"  [{lang}] PRED: {pred}")
            print(f"       REF:  {ref}")
    print()

# ── 11. Save results to JSON ──────────────────────────────────────────────────
out_path = "outputs/xm3600_results.json"
os.makedirs("outputs", exist_ok=True)
# Remove preds/refs from saved JSON (large)
save_results = {
    lang: {k: v for k, v in r.items() if k not in ("preds", "refs")}
    for lang, r in results.items()
}
save_results["_meta"] = {
    "checkpoint":    args.checkpoint,
    "langs":         args.langs,
    "n_images":      len(eval_image_ids),
    "training_data": "English Flickr8k only",
    "note":          "Non-English = zero-shot forced generation",
}
with open(out_path, "w") as f:
    json.dump(save_results, f, indent=2)
print(f"\nResults saved to {out_path}")