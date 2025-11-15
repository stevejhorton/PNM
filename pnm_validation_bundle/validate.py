#!/usr/bin/env python3
"""
PNM validation – tiny MLP default, transformer optional
Tested on M2 Mac:  python validate.py --model transformer --layers 6 --heads 8 --attack all
"""
import argparse, json, time, torch
from tqdm import tqdm
from models import MLP, Transformer
from pnm_core import ParityNode, MasterNode

# ---------- attacks (no leaf in-place) ----------
def random_flip(model, n=1000, scale=0.1):
    with torch.no_grad():
        for _ in range(n):
            p = list(model.parameters())[torch.randint(len(list(model.parameters())), (1,)).item()]
            idx = tuple(torch.randint(0, s, (1,)).item() for s in p.shape)
            p[idx] = p[idx] + torch.randn(1).item() * scale   # not in-place on view

def rank1(model, scale=0.05):
    with torch.no_grad():
        w = model.head.weight
        u = torch.randn(w.shape[0], 1, device=w.device)
        v = torch.randn(1, w.shape[1], device=w.device)
        w.copy_(w + scale * (u @ v))          # copy_ avoids leaf view issue

def adaptive(model, canary_x, canary_y, max_flips=5000, eps=1e-4):
    model.eval()
    with torch.no_grad():
        orig = model(canary_x)
        changed = 0
        for _ in range(max_flips):
            p = list(model.parameters())[torch.randint(len(list(model.parameters())), (1,)).item()]
            idx = tuple(torch.randint(0, s, (1,)).item() for s in p.shape)
            delta = torch.randn(1).item() * 0.01
            old = p[idx].clone()
            p[idx] = p[idx] + delta
            if not torch.allclose(model(canary_x), orig, atol=eps):
                p[idx] = old          # revert
            else:
                changed += 1
        print(f'adaptive: flipped {changed} coords while keeping canaries within {eps}')

# ---------- core helpers ----------
def inject_pnm(model, density=0.005, fan_in=4, key=None, max_nodes=5000):
    if key is None:
        key = torch.rand(32).numpy().tobytes()
    all_p = [p for p in model.parameters() if p.requires_grad and p.numel() > 0]
    n_weights = sum(p.numel() for p in all_p)
    n_nodes = min(int(n_weights * density), max_nodes)
    print(f'Building {n_nodes} parity nodes (density cap {max_nodes})...')
    pnodes = []
    for i in tqdm(range(n_nodes), desc='parity nodes', unit=' nd'):
        tensor = all_p[torch.randint(len(all_p), (1,)).item()]
        pnodes.append(ParityNode(f'pn{i}', [tensor], fan_in))

    out_ids = {id(p) for p in ([model.head.weight, model.head.bias] if hasattr(model, 'head') else [])}
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
    args = parser.parse_args()

    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # build model
    if args.model == 'transformer':
        model = Transformer(layers=args.layers, nhead=args.heads).to(device)
        density = 0.005
    else:
        model = MLP().to(device)
        density = 0.01
    # PNM overlay
    pdict, masters, key = inject_pnm(model, density=density)
    # canary set
    canary_x = torch.randint(0, 1000, (64, 20)).to(device)
    with torch.no_grad():
        canary_y = model(canary_x)

    # attack battery
    attacks = []
    if args.attack in ('all','random'): attacks.append(('random-1k', lambda: random_flip(model, 1000)))
    if args.attack in ('all','rank1'):  attacks.append(('rank1', lambda: rank1(model)))
    if args.attack in ('all','adaptive'): attacks.append(('adaptive', lambda: adaptive(model, canary_x, canary_y)))

    results = {}
    for name, atk in attacks:
        atk()
        ok, _ = bench(model, pdict, masters, canary_x, canary_y)
        results[name] = ok
        print(f'{name:12} passed_verify={ok}')
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
