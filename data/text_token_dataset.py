# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset

import open_clip
from PIL import Image

from ultralytics.utils import LOGGER


class TextTokenDataset(Dataset):
    """
    Image + Text Token Dataset (CLIP-based)

    TSV format (space-separated text part):
        image_id <TAB> count=high size=medium region=top-center

    Example:
        000     count=high size=medium region=top-center
    """

    def __init__(
        self,
        img_dir: str,
        tsv_path: str,
        clip_model_path: str,
        clip_model_name: str = "ViT-B-32",
        device: str | None = None,
        img_suffix: str = ".jpg",
    ):
        self.img_dir = Path(img_dir)
        self.tsv_path = Path(tsv_path)
        self.img_suffix = img_suffix
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        assert self.tsv_path.exists(), f"TSV file not found: {tsv_path}"
        assert Path(clip_model_path).exists(), f"CLIP weights not found: {clip_model_path}"

        # ------------------------------------------------
        # Load TSV
        # ------------------------------------------------
        self.samples: List[Dict] = []
        with open(self.tsv_path, "r") as f:
            for line in f:
                idx, text = line.strip().split("\t")
                self.samples.append(
                    {
                        "id": idx,
                        "text": text,
                        "img_path": self.img_dir / f"{idx}{self.img_suffix}",
                    }
                )

        LOGGER.info(f"[TextTokenDataset] Loaded {len(self.samples)} samples")

        # ------------------------------------------------
        # Load CLIP
        # ------------------------------------------------
        self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=None,
        )

        ckpt = torch.load(clip_model_path, map_location="cpu")
        self.clip_model.load_state_dict(ckpt, strict=False)
        self.clip_model = self.clip_model.to(self.device).eval()

        self.tokenizer = open_clip.get_tokenizer(clip_model_name)

        # Cache text embeddings
        self._build_text_cache()

    # ------------------------------------------------
    # Text Encoding
    # ------------------------------------------------
    def _build_text_cache(self):
        """
        Encode all text prompts once to avoid repeated CLIP forward.
        """
        LOGGER.info("[TextTokenDataset] Encoding text tokens with CLIP...")
        self.text_feats = []

        with torch.no_grad():
            for s in self.samples:
                tokens = self.tokenizer([s["text"]]).to(self.device)
                text_feat = self.clip_model.encode_text(tokens)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                self.text_feats.append(text_feat.squeeze(0).cpu())

        self.text_feats = torch.stack(self.text_feats, dim=0)  # [N, D]

    # ------------------------------------------------
    # PyTorch Dataset API
    # ------------------------------------------------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]

        # Image
        img = Image.open(s["img_path"]).convert("RGB")
        img = self.clip_preprocess(img)

        return {
            "img": img,
            "text_feats": self.text_feats[idx],
        }
