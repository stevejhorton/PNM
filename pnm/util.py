from __future__ import annotations
import torch, hmac, hashlib, json
from typing import Iterable

def tensor_sum(tensors: Iterable[torch.Tensor]) -> float:
    flat = [t.view(-1) for t in tensors]
    if not flat:
        return 0.0
    return float(torch.cat(flat, dim=0).sum().item())

def hmac_hex(key: bytes, data: bytes, algo: str = 'sha256') -> str:
    return hmac.new(key, data, getattr(hashlib, algo)).hexdigest()

def serialize_ordered(obj) -> bytes:
    # stable JSON serialization
    return json.dumps(obj, sort_keys=True, separators=(',', ':')).encode('utf-8')
