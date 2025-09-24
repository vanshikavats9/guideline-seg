from __future__ import annotations
import json, pickle, os
from pathlib import Path
from typing import List, Dict, Sequence


import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image

# --- Enricher ---
class Enricher:
    def __init__(self, device: str | None = None, st_device: str | None = None):
        self.gemma_device = device or ("cuda:1" if torch.cuda.is_available() else "cpu")
        self.st_device = st_device or self.gemma_device
        self._st = None
        self._gemma_processor = None
        self._gemma_model = None
        self._gemma_id = "google/gemma-3-4b-it"
        self.iter_count = 0
        self.cleanup_frequency = 20 

    def _load_st_model(self):
        if self._st is None:
            self._st = SentenceTransformer("all-MiniLM-L6-v2", device=self.st_device)

    def _load_gemma_model(self):
        if self._gemma_model is None:
            self._gemma_processor = AutoProcessor.from_pretrained(self._gemma_id)
            self._gemma_model = (
                Gemma3ForConditionalGeneration
                .from_pretrained(self._gemma_id, torch_dtype=torch.bfloat16)
                .to(self.gemma_device)
                .eval()
            )

    def _cleanup_if_needed(self):
        self.iter_count += 1
        if self.iter_count >= self.cleanup_frequency:
            print(f"Performing cleanup after {self.cleanup_frequency} iterations...")
            self._cleanup()
            self.iter_count = 0
            for device in [self.gemma_device, self.st_device]:
                if device.startswith("cuda"):
                    with torch.cuda.device(device.split(":")[1]):
                        torch.cuda.empty_cache()

    def _cleanup(self):
        if self._st is not None:
            del self._st
            self._st = None
        if self._gemma_model is not None:
            del self._gemma_model
            self._gemma_model = None
        if self._gemma_processor is not None:
            del self._gemma_processor
            self._gemma_processor = None

    def build_query(self, image: Image.Image, user_prompt: str) -> str:
        caption = self._caption(image)
        w, h = image.size
        query = (
            f"Task: {user_prompt}. "
            f"Caption: {caption}. "
            f"Resolution: {w}x{h}."
        )
        print("query: ", query)
        self._cleanup_if_needed()
        return query

    def embed(self, texts: List[str]) -> torch.Tensor:
        self._load_st_model()
        result = self._st.encode(texts, convert_to_tensor=True)
        self._cleanup_if_needed()
        return result

    @torch.inference_mode()
    def _caption(self, img: Image.Image) -> str:
        try:
            self._load_gemma_model()
            class_name = "pedestrian"
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You describe images. Be concise but detailed."}]},
                {"role": "user",  "content": [
                    {"type": "image", "image": img},
                    {"type": "text",  "text": f"If a '{class_name}' is visible, what is it doing and just list the objects related to it."
                                                f"Two sentences max. Only the description."}
                ]}
            ]

            inputs = self._gemma_processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.gemma_device, dtype=torch.bfloat16)

            input_len = inputs["input_ids"].shape[-1]
            gen_ids = self._gemma_model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
            )
            gen_ids = gen_ids[0][input_len:]
            caption = self._gemma_processor.decode(gen_ids, skip_special_tokens=True)
            return caption

        except Exception as e:
            print("Gemma 3 caption failed:", e)
            return "an image"


# --- Retriever ---
class Retriever:
    _guidelines_path = Path("instructions/waymo/waymo_guidelines.json")
    _idx_path        = Path("instructions/waymo/waymo_guidelines.index")
    _meta_path       = Path("instructions/waymo/waymo_guidelines_meta.pkl")

    def __init__(self, device: str | None = None, dim: int = 384):
        self.device = device or ("cuda:2" if torch.cuda.is_available() else "cpu")
        self.enricher = Enricher(st_device=self.device)  # Only use ST model
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.meta = []
        self._load_or_build_index()

    def _load_or_build_index(self):
        if self._idx_path.exists() and self._meta_path.exists():
            self._load_index()
        else:
            self._build_index()
            self._save_index()

    def fetch(self, query: str, k: int = 10) -> List[Dict]:
        q_vec = self.enricher.embed([query]).cpu().numpy().astype("float32")
        scores, ids = self.index.search(q_vec, k)
        hits = []
        for idx in ids[0]:
            if idx == -1:
                continue
            hits.append(self.meta[idx])
        return hits

    def _save_index(self) -> None:
        self._idx_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self._idx_path))
        self._meta_path.write_bytes(pickle.dumps(self.meta))

    def _load_index(self) -> None:
        self.index = faiss.read_index(str(self._idx_path))
        self.meta = pickle.loads(self._meta_path.read_bytes())

    # --- build index from scratch ---
    def _build_index(self) -> None:
        with self._guidelines_path.open() as f:
            data = json.load(f)

        base_rules: Sequence[Dict] = data.get("guidelines") or data.get("rules") or []
        self._add_vectors(base_rules, weight=1.0)         # static rules
        # self._add_vectors(RuleStore().all(), weight=0.6)  # learned rules (optional)

    # --- add vectors & metadata ---
    def _add_vectors(self, rules: Sequence[Dict], weight: float) -> None:
        phrases: List[str]  = []
        metas:   List[Dict] = []

        for rule in rules:
            text_field = rule.get("embed_text") or rule.get("description")
            if text_field is None:
                continue
            # ensure iterable
            text_list = text_field if isinstance(text_field, list) else [text_field]
            for phrase in text_list:
                phrases.append(phrase)
                metas.append(rule)

        if not phrases:
            return

        vecs = self.enricher.embed(phrases).cpu().numpy().astype("float32")
        vecs *= weight
        assert vecs.shape[1] == self.dim, "Embedding dimension mismatch"

        self.index.add(vecs)
        self.meta.extend(metas)

    def cleanup(self):
        # only use this when completely done with the Retriever instance
        if hasattr(self, 'enricher'):
            self.enricher._cleanup()
            del self.enricher
        if hasattr(self, 'index'):
            del self.index
        if hasattr(self, 'meta'):
            del self.meta
        if isinstance(self.device, str) and self.device.startswith("cuda"):
            try:
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            except Exception:
                # fallback: extract index as int
                with torch.cuda.device(int(self.device.split(":")[1])):
                    torch.cuda.empty_cache()
        elif isinstance(self.device, int):
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()


