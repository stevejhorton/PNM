import hashlib, hmac, secrets, torch, torch.nn as nn

def _hash(x: bytes) -> bytes:
    return hashlib.sha256(x).digest()

class ParityNode:
    """Passive tag over a weight tensor slice."""
    def __init__(self, name: str, tensors: list, fan_in: int = 4):
        self.name = name
        flat = torch.cat([w.detach().flatten() for w in tensors])
        if flat.numel() < fan_in:                       # tiny tensor → use whole thing
            self.idx = torch.arange(flat.numel())
        else:
            self.idx = torch.randperm(flat.numel())[:fan_in]
        self.vals = flat[self.idx].clone()

    def current_hash(self, tensors: list) -> bytes:
        flat = torch.cat([w.detach().flatten() for w in tensors])
        return _hash(flat[self.idx].numpy().tobytes())
class MasterNode:
    def __init__(self, name: str, pnodes: list, key: bytes):
        self.name, self.key = name, key
        self.pnode_names = [p.name for p in pnodes]
        self.hmac = self._compute(pnodes)

    def _compute(self, pnodes: list) -> bytes:
        payload = b''.join(p.current_hash([]) for p in pnodes)
        return hmac.new(self.key, payload, hashlib.sha256).digest()

    def verify(self, pnodes_dict: dict) -> bool:
        return self.hmac == self._compute([pnodes_dict[n] for n in self.pnode_names])
