#!/usr/bin/env python3
"""
PNM validation – tiny-MLP by default, big transformer optional
python validate.py                          # 111-param MLP
python validate.py --model transformer --layers 6 --heads 8 --attack all
"""
import argparse, json, time, torch, sys
from tqdm import tqdm
from models import MLP, Transformer
from pnm_core import ParityNode, MasterNode
from attacks import random_flip, rank1, adaptive

# ---------- core helpers ----------
def inject_pnm(model, density=0.005, fan_in=4, key=None, max_nodes=5000):
    if key is None:
        key = torch.rand(32).numpy().tobytes()
    all_p = [p for p in model.parameters() if p.requires_grad]
    n_weights = sum(p.numel() for p in all_p)
    n_nodes = min(int(n_weights * density), max_nodes)
    print(f'Building {n_nodes} parity nodes (density cap {max_nodes})...')
    pnodes = []
    for i in tqdm(range(n_nodes), desc='parity nodes', unit=' nd'):
        tensor = all_p[torch.randint(len(all_p), (1,)).item()]
        pnodes.append(ParityNode(f'pn{i}', [tensor], fan_in))

    out_p = [model.head.weight, model.head.bias] if hasattr(model, 'head') else []
    out_ids = {id(p) for p in out_p}
    m_core = MasterNode('m_core', [pn for pn in pnodes if any(id(t) in out_ids for t in pn.vals)], key)
    m_edge = MasterNode('m_edge', pnodes[::2], key)
    masters = [m_core, m_edge] + [MasterNode(f'm{i}', pnodes[i::30], key) for i in range(30)]
    pdict = {pn.name: pn for pn in pnodes}
    return pdict, masters, key

def bench(model, pdict, masters, canary_x, canary_y):
    t0 = time.time()
    ok = all(m.verify(pdict) for m in masters) and torch.allclose(model(canary_x), canary_y, atol=1e-4)
    return ok, (time.time() - t0) * 1000   # ms

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp','transformer'], default='mlp')
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--attack', choices=['all','random','rank1','adaptive'], default='all')
    parser.add_argument('--bench', action='store_true', help='latency micro-bench only')
    args = parser.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # build model
    if args.model == 'transformer':
        model = Transformer(layers=args.layers, nhead=args.heads).to(device)
        pdict, masters, key = inject_pnm(model, density=0.005, max_nodes=5000)
    else:
        model = MLP().to(device)
        pdict, masters, key = inject_pnm(model, density=0.01)
    # PNM overlay
    pdict, masters, key = inject_pnm(model, density=density)
    # canary set
    canary_x = torch.randint(0, 1000, (64, 20)).to(device)
    with torch.no_grad():
        canary_y = model(canary_x)

    if args.bench:
        ok, t = bench(model, pdict, masters, canary_x, canary_y)
        print(f'full_verify {t:.2f} ms, passed={ok}')
        return

    # attack battery
    attacks = []
    if args.attack in ('all','random'): attacks.append(('random-1k', lambda: random_flip(model, 1000)))
    if args.attack in ('all','rank1'):  attacks.append(('rank1', lambda: rank1(model)))
    if args.attack in ('all','adaptive'): attacks.append(('adaptive', lambda: adaptive(model, canary_x, canary_y)))

    results = {}
    for name, atk in attacks:
        # restore clean weights
        model.load_state_dict(model.state_dict())
        atk()
        ok, _ = bench(model, pdict, masters, canary_x, canary_y)
        results[name] = ok
        print(f'{name:12} passed_verify={ok}')
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
