from __future__ import annotations
import torch, json, random, pathlib, math, hashlib, hmac
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .util import tensor_sum, hmac_hex, serialize_ordered

@dataclass
class ParityNodeSpec:
    id: str
    layer_name: str
    indices: List[Tuple[int, int]]  # list of (row,col) in the layer weight matrix

@dataclass
class MasterSpec:
    id: str
    pn_ids: List[str]

@dataclass
class LockedMap:
    pn_values: Dict[str, float]
    mn_hmacs: Dict[str, str]
    root_hmac: str
    algo: str = "sha256"

class HPNMManager:
    def __init__(self, model: torch.nn.Module, algo: str = "sha256"):
        self.model = model
        self.algo = algo
        self.pn_specs: List[ParityNodeSpec] = []
        self.mn_specs: List[MasterSpec] = []
        self.fixed_core_master: str | None = None
        self.fixed_edge_master: str | None = None

    # ----- Placement ---------------------------------------------------------
    def _get_layer_weights(self):
        layers = []
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layers.append((name, module.weight))
        return layers

    def place_nodes(self, num_parity: int = 100, seed: int = 0, density: float = 0.5,
                    topk_fraction: float = 0.3):
        random.seed(seed)
        torch.manual_seed(seed)
        layers = self._get_layer_weights()
        if not layers:
            raise RuntimeError("No Linear layers found for parity placement.")

        # Build candidate index lists
        all_indices = []
        for lname, W in layers:
            rows, cols = W.shape
            for r in range(rows):
                for c in range(cols):
                    all_indices.append((lname, r, c))

        if not all_indices:
            raise RuntimeError("No weight indices found.")

        # Deterministic sample based on density/num_parity
        target = min(num_parity, max(1, int(len(all_indices) * density)))
        picks = random.sample(all_indices, target)

        # Group picks by layer and create PN specs with small fan-in (2..10)
        pid = 0
        layer_groups: Dict[str, List[Tuple[int,int]]] = {}
        for lname, r, c in picks:
            layer_groups.setdefault(lname, []).append((r, c))

        self.pn_specs.clear()
        for lname, idxs in layer_groups.items():
            # chunk indices into random fan-ins
            i = 0
            while i < len(idxs):
                fanin = random.randint(2, min(10, len(idxs) - i))
                chunk = idxs[i:i+fanin]
                pid += 1
                self.pn_specs.append(ParityNodeSpec(id=f"PN{pid}", layer_name=lname, indices=chunk))
                i += fanin

        # Masters: fixed core (output-layer PNs) and edge (top-activation region PNs)
        out_layer = layers[-1][0]
        core_pns = [p.id for p in self.pn_specs if p.layer_name == out_layer]
        edge_pns = [p.id for p in self.pn_specs if p.layer_name != out_layer]

        if not core_pns:
            anypn = self.pn_specs[0].id
            core_pns = [anypn]
            edge_pns = [p.id for p in self.pn_specs if p.id != anypn]

        self.fixed_core_master = "MN_core"
        self.fixed_edge_master = "MN_edge"
        self.mn_specs = [
            MasterSpec(id=self.fixed_core_master, pn_ids=core_pns),
            MasterSpec(id=self.fixed_edge_master, pn_ids=edge_pns),
        ]

    # ----- Locking -----------------------------------------------------------
    def _pn_value(self, spec: ParityNodeSpec) -> float:
        # sum the selected weights in that layer
        W = dict((name, mod.weight) for name, mod in self.model.named_modules() if isinstance(mod, torch.nn.Linear))[spec.layer_name]
        tensors = [W[r, c] for (r, c) in spec.indices]
        return float(torch.stack(tensors).sum().item())

    def lock(self, key_m: bytes, key_root: bytes, out_path: str = ".artifacts"):
        p = pathlib.Path(out_path); p.mkdir(parents=True, exist_ok=True)
        pn_values = {spec.id: self._pn_value(spec) for spec in self.pn_specs}

        # Master HMACs
        mn_hmacs = {}
        for mspec in self.mn_specs:
            entries = [(pid, pn_values.get(pid, 0.0)) for pid in mspec.pn_ids]
            payload = serialize_ordered(entries)
            mn_hmacs[mspec.id] = hmac_hex(key_m, payload, self.algo)

        # Root HMAC
        masters_sorted = sorted(mn_hmacs.items())  # (id, hmac)
        root_payload = serialize_ordered(masters_sorted)
        root_h = hmac_hex(key_root, root_payload, self.algo)

        locked = LockedMap(pn_values=pn_values, mn_hmacs=mn_hmacs, root_hmac=root_h, algo=self.algo)
        (p / "hp_parity_map.json").write_text(json.dumps({
            "pn_values": locked.pn_values,
            "mn_hmacs": locked.mn_hmacs,
            "root_hmac": locked.root_hmac,
            "algo": locked.algo
        }, indent=2))

        # Publish a public hash of the serialized locked map (chain-of-trust)
        public_hash = hashlib.blake2b(json.dumps({
            "pn_values": locked.pn_values,
            "mn_hmacs": locked.mn_hmacs,
            "root_hmac": locked.root_hmac,
            "algo": locked.algo
        }, sort_keys=True).encode("utf-8"), digest_size=32).hexdigest()
        (p / "public_hash.txt").write_text(public_hash)

        # Also persist placement specs for reproducibility
        (p / "placement.json").write_text(json.dumps({
            "pn_specs": [spec.__dict__ for spec in self.pn_specs],
            "mn_specs": [dict(id=ms.id, pn_ids=ms.pn_ids) for ms in self.mn_specs],
            "fixed_core": self.fixed_core_master,
            "fixed_edge": self.fixed_edge_master
        }, indent=2))

        return locked, public_hash

    # ----- Verification ------------------------------------------------------
    def _recompute_pn_values(self) -> Dict[str, float]:
        return {spec.id: self._pn_value(spec) for spec in self.pn_specs}

    def fast_check(self, key_m: bytes, locked_map_path: str = ".artifacts/hp_parity_map.json") -> bool:
        locked = json.loads(pathlib.Path(locked_map_path).read_text())
        pn_values = self._recompute_pn_values()

        # Verify the two fixed masters
        for mid in (self.fixed_core_master, self.fixed_edge_master):
            mspec = next(m for m in self.mn_specs if m.id == mid)
            entries = [(pid, pn_values.get(pid, 0.0)) for pid in mspec.pn_ids]
            payload = serialize_ordered(entries)
            recomputed = hmac_hex(key_m, payload, locked.get("algo","sha256"))
            if recomputed != locked["mn_hmacs"].get(mid, ""):
                return False
        return True

    def full_check(self, key_m: bytes, key_root: bytes, locked_map_path: str = ".artifacts/hp_parity_map.json") -> bool:
        locked = json.loads(pathlib.Path(locked_map_path).read_text())
        pn_values = self._recompute_pn_values()

        mn_hmacs = {}
        for mspec in self.mn_specs:
            entries = [(pid, pn_values.get(pid, 0.0)) for pid in mspec.pn_ids]
            payload = serialize_ordered(entries)
            mn_hmacs[mspec.id] = hmac_hex(key_m, payload, locked.get("algo","sha256"))

        # Root
        masters_sorted = sorted(mn_hmacs.items())
        root_payload = serialize_ordered(masters_sorted)
        root_h = hmac_hex(key_root, root_payload, locked.get("algo","sha256"))

        return (mn_hmacs == locked["mn_hmacs"]) and (root_h == locked["root_hmac"])
