"""Model descriptors & metadata catalog for installed Ollama models.

Implements enumerations and a catalog of known locally-present models with
inferred attributes (family, domain, size category, quantization, variant).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List
from datetime import datetime

class ModelFamily(Enum):
    LLAMA = "LLaMA"
    CODE_LLAMA = "CodeLLaMA"
    TINY_LLAMA = "TinyLLaMA"
    QWEN = "Qwen"
    QWEN_CODER = "QwenCoder"
    MISTRAL = "Mistral"
    GEMMA = "Gemma"
    PHI = "Phi"
    DEEPSEEK = "DeepSeek"
    DEEPSEEK_CODER = "DeepSeekCoder"

class Domain(Enum):
    GENERAL = "General"
    CODING = "Coding"
    LIGHTWEIGHT = "Lightweight"

class SizeCategory(Enum):
    SMALL = "Small"        # ≤ 3B
    MEDIUM = "Medium"      # 4B–16B
    LARGE = "Large"        # 30B–70B+

class Quantization(Enum):
    Q2_K = "Q2_K"
    Q3_K_M = "Q3_K_M"
    Q4_K_M = "Q4_K_M"
    Q4_0 = "Q4_0"
    FULL_PRECISION = "FullPrecision"

@dataclass(frozen=True)
class ModelDescriptor:
    name: str
    family: ModelFamily
    purpose: Domain
    size: SizeCategory
    quant: Quantization
    disk_size_gb: Optional[float]
    variant: str
    installed_at: Optional[datetime] = None

# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def _infer_variant(tag: str) -> str:
    lower = tag.lower()
    if "lite" in lower and "instruct" in lower:
        return "lite-instruct"
    for token in ("instruct", "mini", "latest"):
        if token in lower:
            return token
    return tag

def _size_category_from_disk(size_gb: Optional[float]) -> SizeCategory:
    if size_gb is None:
        return SizeCategory.MEDIUM
    if size_gb <= 3.0:
        return SizeCategory.SMALL
    if size_gb <= 16.0:
        return SizeCategory.MEDIUM
    return SizeCategory.LARGE

def _quant_from_name(name: str) -> Quantization:
    lowered = name.lower()
    if "q2_k" in lowered:
        return Quantization.Q2_K
    if "q3_k_m" in lowered:
        return Quantization.Q3_K_M
    if "q4_k_m" in lowered:
        return Quantization.Q4_K_M
    if "q4_0" in lowered or "q4-0" in lowered:
        return Quantization.Q4_0
    return Quantization.FULL_PRECISION

def _family_from_name(name: str) -> ModelFamily:
    base = name.split(":", 1)[0].lower()
    if base.startswith("llama"):
        return ModelFamily.LLAMA
    if base.startswith("codellama"):
        return ModelFamily.CODE_LLAMA
    if base.startswith("tinyllama"):
        return ModelFamily.TINY_LLAMA
    if base.startswith("qwen2.5-coder") or base.startswith("qwen2.5_coder") or base.startswith("qwen2-coder"):
        return ModelFamily.QWEN_CODER
    if base.startswith("qwen"):
        return ModelFamily.QWEN
    if base.startswith("mistral"):
        return ModelFamily.MISTRAL
    if base.startswith("gemma"):
        return ModelFamily.GEMMA
    if base.startswith("phi"):
        return ModelFamily.PHI
    if base.startswith("deepseek-coder-v2") or base.startswith("deepseek-coder"):
        return ModelFamily.DEEPSEEK_CODER
    if base.startswith("deepseek"):
        return ModelFamily.DEEPSEEK
    return ModelFamily.LLAMA

def _domain_from_family(fam: ModelFamily) -> Domain:
    if fam in {ModelFamily.CODE_LLAMA, ModelFamily.DEEPSEEK_CODER, ModelFamily.QWEN_CODER}:
        return Domain.CODING
    if fam in {ModelFamily.TINY_LLAMA, ModelFamily.PHI}:
        return Domain.LIGHTWEIGHT
    return Domain.GENERAL

def _build_descriptor(name: str, disk_size_gb: Optional[float]) -> ModelDescriptor:
    quant = _quant_from_name(name)
    family = _family_from_name(name)
    purpose = _domain_from_family(family)
    tag = name.split(":", 1)[1] if ":" in name else name
    variant = _infer_variant(tag)
    size_category = _size_category_from_disk(disk_size_gb)
    return ModelDescriptor(
        name=name,
        family=family,
        purpose=purpose,
        size=size_category,
        quant=quant,
        disk_size_gb=disk_size_gb,
        variant=variant,
        installed_at=None,
    )

# Raw model list with approximate disk sizes (GB)
_RAW_MODELS: Dict[str, Optional[float]] = {
    "llama3.3:70b-instruct-q2_K": 26.0,
    "llama3.3:70b-instruct-q3_K_M": 34.0,
    "qwen2.5:3b-instruct-q4_K_M": 1.9,
    "qwen2.5:7b-instruct-q4_K_M": 4.7,
    "gemma:7b-instruct-q4_K_M": 5.5,
    "phi3:mini": 2.2,
    "codellama:13b-instruct": 7.4,
    "mistral:7b-instruct-q4_K_M": 4.4,
    "deepseek-coder:6.7b": 3.8,
    "deepseek-coder-v2:16b-lite-instruct-q4_K_M": 10.0,
    "qwen2.5-coder:7b": 4.7,
    "mistral:7b-instruct": 4.4,
    "codellama:13b-instruct-q4_K_M": 7.9,
    "tinyllama:latest": 0.637,
    "mistral:latest": 4.4,
    "codellama:13b-instruct-q4_0": None,
}

MODEL_CATALOG: Dict[str, ModelDescriptor] = {name: _build_descriptor(name, size) for name, size in _RAW_MODELS.items()}

def describe_model(name: str) -> Optional[ModelDescriptor]:
    return MODEL_CATALOG.get(name)

def list_models() -> List[ModelDescriptor]:
    return list(MODEL_CATALOG.values())

__all__ = [
    "ModelFamily",
    "Domain",
    "SizeCategory",
    "Quantization",
    "ModelDescriptor",
    "MODEL_CATALOG",
    "describe_model",
    "list_models",
]
