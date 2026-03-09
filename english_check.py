"""
Task 1 — Projection Layer Training (English Sanity Check)

Architecture:
  BLIP-2 ViT + Q-Former  → frozen, precompute features then delete
  ProjectionMLP 768→768  → trained from scratch in fp16
  mT5-base               → frozen, fp16
"""

import gc
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Blip2Processor,
    Blip2Model,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)
from datasets import load_dataset
from tqdm import tqdm

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    total_vram = props.total_memory / 1e9
    print(f"GPU: {props.name} | Total VRAM: {total_vram:.1f} GB")

def vram_used():
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved(0) / 1e9
    return 0.0

def tensor_stats(t, name="tensor"):
    """Print min/max/mean/std and flag NaN or Inf."""
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    print(f"  {name}: shape={list(t.shape)} dtype={t.dtype} "
          f"min={t.min():.4f} max={t.max():.4f} "
          f"mean={t.mean():.4f} std={t.std():.4f} "
          f"{'NaN! ✗' if has_nan else ''}"
          f"{'Inf! ✗' if has_inf else ''}"
          f"{'✓' if not has_nan and not has_inf else ''}")
    return has_nan, has_inf

# ── 1. Load Flickr8k ──────────────────────────────────────────────────────────
print("\n── 1. Load Flickr8k " + "─"*50)
os.environ["HF_DATASETS_OFFLINE"] = "1"
flickr = load_dataset("jxie/flickr8k", split="train")
print(f"Loaded: {len(flickr)} samples | Columns: {flickr.column_names}")
print(f"Example caption: {flickr[0]['caption_0']}")

# ── 2. Precompute visual features ─────────────────────────────────────────────
print("\n── 2. Precompute Q-Former Features " + "─"*35)
FEATURE_FILE = "flickr_features.pt"

if os.path.exists(FEATURE_FILE):
    print(f"Found {FEATURE_FILE} — loading from cache.")
    saved        = torch.load(FEATURE_FILE, weights_only=False)
    all_features = saved["features"]
    captions     = saved["captions"]
    print(f"  Loaded {len(captions)} feature vectors")

else:
    print("Loading BLIP-2 blip2-opt-2.7b (ViT + Q-Former)...")
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

    images   = [s["image"]     for s in flickr]
    captions = [s["caption_0"] for s in flickr]

    all_features  = []
    PRECOMPUTE_BS = 32
    print(f"  Precomputing features for {len(images)} images (batch={PRECOMPUTE_BS})...")

    for i in tqdm(range(0, len(images), PRECOMPUTE_BS)):
        batch_imgs = images[i : i + PRECOMPUTE_BS]
        inputs = blip2_processor(images=batch_imgs, return_tensors="pt").to(device)

        with torch.no_grad():
            vision_out = blip2_model.vision_model(
                pixel_values=inputs.pixel_values.half(),
                return_dict=True,
            )
            image_embeds = vision_out.last_hidden_state          # (B, 257, 1408)
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
            feats = qformer_out.last_hidden_state                # (B, 32, 768)

        all_features.append(feats.cpu().float())

        # Debug: check first batch before continuing
        if i == 0:
            print("  First batch feature check:")
            tensor_stats(feats.cpu().float(), "Q-Former output [batch 0]")

    all_features = torch.cat(all_features, dim=0)               # (N, 32, 768)
    print(f"  Features shape: {all_features.shape}")

    torch.save({"features": all_features, "captions": captions}, FEATURE_FILE)
    print(f"  Saved: {FEATURE_FILE}")

    del blip2_model, blip2_processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  BLIP-2 deleted | VRAM free: {total_vram - vram_used():.1f} GB")

# ── Feature health check ──────────────────────────────────────────────────────
print("\nFeature health check:")
has_nan, has_inf = tensor_stats(all_features, "all_features")
assert not has_nan, "NaNs in precomputed features — cannot continue"
assert not has_inf, "Infs in precomputed features — cannot continue"

# ── 3. Load mT5-base ──────────────────────────────────────────────────────────
print("\n── 3. Load mT5-base (frozen, fp16) " + "─"*35)
os.environ["TRANSFORMERS_OFFLINE"] = "1"
mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-base")
mt5_model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/mt5-base"
).to(device)
for param in mt5_model.parameters():
    param.requires_grad = False
mt5_model.eval()
print(f"  mT5-base loaded (fp16, frozen) | VRAM: {vram_used():.1f} GB")

# Quick tokenizer check
tok_test = mt5_tokenizer("a dog running in the park", return_tensors="pt")
print(f"  Tokenizer check — ids shape: {tok_test.input_ids.shape} ✓")

# ── 4. Projection MLP ─────────────────────────────────────────────────────────
print("\n── 4. Projection MLP " + "─"*49)

class ProjectionMLP(nn.Module):
    """
    Maps Q-Former output (B, 32, 768) → mT5 encoder input space (B, 32, 768).
    Runs in fp16 natively — avoids the fp32→fp16 cast that caused NaN.
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
'''

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int = 768, out_dim: int = 768):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.act     = nn.GELU()
        self.norm    = nn.LayerNorm(out_dim)   # stays fp32
        self.linear2 = nn.Linear(out_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.act(x)
        x = self.norm(x.float()).to(x.dtype)   # cast to fp32 for norm, back to fp16
        x = self.linear2(x)
        return x
'''
'''
projection = ProjectionMLP().to(device).to(torch.float16)
projection.norm.float() 
n_params   = sum(p.numel() for p in projection.parameters())
print(f"  Trainable params: {n_params:,}")
print(f"  dtype: {next(projection.parameters()).dtype}")
'''
projection = ProjectionMLP().to(device)   # fp32
n_params   = sum(p.numel() for p in projection.parameters())
print(f"  Trainable params: {n_params:,}")
print(f"  dtype: {next(projection.parameters()).dtype}")

# Dry-run: confirm shapes and no NaN at init
with torch.no_grad():
    #dummy     = torch.zeros(2, 32, 768, dtype=torch.float16, device=device)
    dummy = torch.zeros(2, 32, 768, dtype=torch.float32, device=device)

    dummy_out = projection(dummy)
    print(f"  Dry-run: in={list(dummy.shape)} → out={list(dummy_out.shape)}")
    tensor_stats(dummy_out, "projection(zeros)")

# ── 5. Dataset ────────────────────────────────────────────────────────────────
print("\n── 5. Dataset " + "─"*56)

class PrecomputedDataset(Dataset):
    def __init__(self, features, captions, tokenizer, max_length=64):
        self.features   = features
        self.captions   = captions
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        feature = self.features[idx]  # (32, 768) float32 — cast to fp16 in train loop
        labels  = self.tokenizer(
            self.captions[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        return feature, labels


full_dataset = PrecomputedDataset(all_features, captions, mt5_tokenizer)
train_size   = int(0.9 * len(full_dataset))
val_size     = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                          num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=16, shuffle=False,
                          num_workers=2, pin_memory=True)
print(f"  Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ── Sample caption preview ────────────────────────────────────────────────────
PREVIEW_INDICES = val_dataset.indices[:5]  # fixed across epochs for comparability

def preview_captions(epoch_label="untrained"):
    """Generate captions for 5 fixed val images and print pred vs reference."""
    projection.eval()
    print(f"\n  ── Caption preview [{epoch_label}] " + "─"*28)
    for rank, idx in enumerate(PREVIEW_INDICES):
        feature = all_features[idx].unsqueeze(0).to(device)
        ref     = flickr[idx]["caption_0"]
        with torch.no_grad():
            projected = projection(feature)
            out = mt5_model.generate(
                inputs_embeds=projected,
                max_new_tokens=50,
                num_beams=4,
            )
        pred = mt5_tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"  [{rank+1}] PRED: {pred}")
        print(f"       REF:  {ref}")
    print()

# ── 6. Training ───────────────────────────────────────────────────────────────
print("\n── 6. Training " + "─"*55)

def train_epoch(loader, optimizer):
    projection.train()
    total_loss  = 0.0
    total_gnorm = 0.0

    for batch_i, (features, labels) in enumerate(tqdm(loader, desc="Train", leave=False)):
        features = features.to(device)
        labels   = labels.to(device)
        projected = projection(features)                     # (B, 32, 768) fp16

        labels_for_loss = labels.clone()
        labels_for_loss[labels_for_loss == mt5_tokenizer.pad_token_id] = -100

        loss = mt5_model(inputs_embeds=projected, labels=labels_for_loss).loss

        # NaN guard — fail fast with diagnostics
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n  ✗ NaN/Inf loss at batch {batch_i}")
            tensor_stats(features,  "  features")
            tensor_stats(projected, "  projected")
            raise RuntimeError("NaN/Inf loss — stopping. Check diagnostics above.")

        optimizer.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(projection.parameters(), max_norm=1.0)
        total_gnorm += gnorm.item()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader), total_gnorm / len(loader)


@torch.no_grad()
def val_epoch(loader):
    projection.eval()
    total_loss = 0.0
    for features, labels in tqdm(loader, desc="Val", leave=False):
        features  = features.to(device)
        labels    = labels.to(device)
        projected = projection(features)

        labels_for_loss = labels.clone()
        labels_for_loss[labels_for_loss == mt5_tokenizer.pad_token_id] = -100

        loss = mt5_model(inputs_embeds=projected, labels=labels_for_loss).loss
        total_loss += loss.item()
    return total_loss / len(loader)


EPOCHS    = 10
optimizer = torch.optim.AdamW(projection.parameters(), lr=3e-4, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_val_loss = float("inf")
os.makedirs("outputs", exist_ok=True)

# Baseline before any training
preview_captions(epoch_label="epoch 0 — untrained baseline")

for epoch in range(EPOCHS):
    train_loss, gnorm = train_epoch(train_loader, optimizer)
    val_loss          = val_epoch(val_loader)
    scheduler.step()
    lr_now = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch+1:2d}/{EPOCHS} | "
          f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
          f"GradNorm: {gnorm:.3f} | lr: {lr_now:.2e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(projection.state_dict(), "outputs/projection_best.pt")
        print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")

    # Preview every 2 epochs and final epoch
    if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
        preview_captions(epoch_label=f"epoch {epoch+1}")

print("\nTraining complete.")

# ── 7. Evaluation — CIDEr and BLEU ───────────────────────────────────────────
print("\n── 7. Evaluation " + "─"*53)
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu

projection.load_state_dict(torch.load("outputs/projection_best.pt", weights_only=True))
projection.eval()

preds_dict  = {}
refs_dict   = {}
EVAL_BATCH  = 16
val_indices = val_dataset.indices

print(f"Generating captions for {len(val_indices)} val samples...")
for start in tqdm(range(0, len(val_indices), EVAL_BATCH)):
    batch_idx = val_indices[start : start + EVAL_BATCH]
    features  = torch.stack(
        [all_features[i] for i in batch_idx]
    ).to(device)
    #features = torch.stack([all_features[i] for i in batch_idx]).to(device)  # stay fp32

    with torch.no_grad():
        projected = projection(features)
        out = mt5_model.generate(
            inputs_embeds=projected,
            max_new_tokens=50,
            num_beams=4,
        )

    for k, idx in enumerate(batch_idx):
        pred = mt5_tokenizer.decode(out[k], skip_special_tokens=True)
        refs = [flickr[idx][f"caption_{j}"] for j in range(5)]
        global_i             = start + k
        preds_dict[global_i] = [pred]
        refs_dict[global_i]  = refs

cider_scorer   = Cider()
cider_score, _ = cider_scorer.compute_score(refs_dict, preds_dict)
bleu_scorer    = Bleu(4)
bleu_scores, _ = bleu_scorer.compute_score(refs_dict, preds_dict)

print(f"\nResults on {len(val_indices)} val images:")
print(f"  CIDEr:  {cider_score * 100:.2f}  (SOTA ~145, reasonable sanity check >20)")
for i, s in enumerate(bleu_scores):
    print(f"  BLEU-{i+1}: {s * 100:.2f}")

# 10 qualitative samples from the full eval set
print("\nQualitative sample (10 random from eval set):")
import random
random.seed(0)
sample_keys = random.sample(list(preds_dict.keys()), min(10, len(preds_dict)))
for key in sample_keys:
    print(f"  PRED: {preds_dict[key][0]}")
    print(f"  REF:  {refs_dict[key][0]}")
    print()

print("Done. Outputs in outputs/")