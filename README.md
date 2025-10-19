# Hardened Parity Network Maps (HPNM) — Reference Demo

This repo contains a minimal, runnable reference implementation of **Hardened Parity Network Maps** with:
- **HMAC-anchored masters** (no linear-sum exploits)
- **Fixed dual-master fast check** (core/output anchored + top-activation anchored)
- **Overlap placement** (targeted + randomized parity nodes)
- **Behavioral canaries** (semantic verification)
- **Demo MLP + attack scripts**

> This is a small demo intended for reproducibility and easy experimentation. It is **not** production code.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 1) Train and lock
python scripts/train_and_lock.py --seed 0 --density 0.5 --num-parity 100 --num-masters 8

# 2) Verify (fast + full) and run attacks
python scripts/verify_and_attack.py --attack none
python scripts/verify_and_attack.py --attack at_rest --m 20
python scripts/verify_and_attack.py --attack runtime --m 20
python scripts/verify_and_attack.py --attack low_rank --scale 0.01
python scripts/verify_and_attack.py --attack random_small --m 100
python scripts/verify_and_attack.py --attack unmonitored_bulk --m 50
```

Artifacts are stored under `.artifacts/`:
- `model.pt` — trained demo MLP
- `hp_parity_map.json` — locked node parities, master HMACs, root digest
- `public_hash.txt` — published hash of the serialized map
- `canaries.json` — canary inputs and expected outputs

## Design (demo)

- **Parity Nodes (PN):** each summarizes a small set of weights (fan-in 2–10). We use deterministic *sums* in the demo for clarity.
- **Masters (MN):** each stores an HMAC over the `(id || parity)` tuples of its PNs.
- **Root:** HMAC over master digests.
- **Fixed dual-master fast check:** two masters are deterministically assigned to (a) all output-layer PNs and (b) top-activation regions.
- **Canaries:** fixed inputs with expected outputs (tolerance) checked each run.
