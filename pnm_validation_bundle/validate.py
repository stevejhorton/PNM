#!/usr/bin/env python3
"""
PNM validation – lightweight by default
python validate.py               # 111-param MLP  (1 s)
python validate.py --large       # 6-layer 8-head transformer (needs 6 GB)
"""
import argparse, json, time, torch, sys
from tqdm import tqdm
from models import MLP, Transformer
from pnm_core import ParityNode, MasterNode
from attacks import random_flip, rank1, adaptive

# ---------- core helpers ----------
def inject_pnm(model, density=0.005, fan_in=4, key=None):
    if key is None:
        key = torch.rand(32).numpy().tobytes()
    all_p = [p for p in model.parameters() if p.requires_grad]
    n_weights = sum(p.numel() for p in all_p)
    n_nodes = int(n_weights * density)
    # stream creation to keep RAM low
    pnodes = []
    for i in tqdm(range(n_nodes), desc='parity nodes', unit_scale=True, unit=' nd'):
        tensor = all_p[torch.randint(len(all_p), (1,)).item()]
        pnodes.append(ParityNode(f'pn{i}', [tensor], fan_in))
    # 2 fixed masters
    out_p = [model.head.weight, model.head.bias] if hasattr(model, 'head') else []
    m_core = MasterNode('m_core', [pn for pn in pnodes if any(t in out_p for t in pn.vals)], key)
    m_edge = MasterNode('m_edge', pnodes[::2], key)  # dummy split
    masters = [m_core, m_edge] + [MasterNode(f'm{i}', pnodes[i::30], key) for i in range(30)]
    pdict = {pn.name: pn for pn in pnodes}
    return pdict, masters, key

def bench(model, pdict, masters, canary_x, canary_y):
    t0 = time.time()
    ok = all(m.verify(pdict) for m in masters) and torch.allclose(model(canary_x), canary_y, atol=1e-4)
    return ok, (time.time() - t0) * 1000   # ms

# ---------- main ----------
def main(large=False):
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # build model
    if large:
        model = Transformer(layers=6, nhead=8).to(device)
        density = 0.005
    else:
        model = MLP().to(device)
        density = 0.01          # higher density for tiny net
    # PNM overlay
    pdict, masters, key = inject_pnm(model, density=density)
    # canary set
    canary_x = torch.randint(0, 1000, (64, 20)).to(device)
    with torch.no_grad():
        canary_y = model(canary_x)
    # attack battery
    attacks = [
        ('random-1k', lambda: random_flip(model, 1000)),
        ('rank1', lambda: rank1(model)),
        ('adaptive', lambda: adaptive(model, canary_x, canary_y))
    ]
    results = {}
    for name, atk in attacks:
        model.load_state_dict(torch.load('clean.pt') if large else model.state_dict())  # restore clean
        atk()
        ok, _ = bench(model, pdict, masters, canary_x, canary_y)
        results[name] = ok
        print(f'{name:12} passed_verify={ok}')
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--large', action='store_true', help='run 10 M-param transformer (needs 6 GB RAM)')
    args = parser.parse_args()
    main(large=args.large)
