import argparse, time, torch, json
from models import MLP, Transformer
from pnm_core import ParityNode, MasterNode
from attacks import random_flip, rank1, adaptive

def inject_pnm(model, density=0.005, fan_in=4, key=None):
    if key is None: key = torch.rand(32).numpy().tobytes()
    all_p = [p for p in model.parameters() if p.requires_grad]
    n_weights = sum(p.numel() for p in all_p)
    n_nodes = int(n_weights * density)
    pnodes = [ParityNode(f'pn{i}', [all_p[torch.randint(len(all_p),(1,)).item()]], fan_in)
              for i in range(n_nodes)]
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
    elapsed = time.time() - t0
    return ok, elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['mlp','transformer'], default='transformer')
    parser.add_argument('--layers', type=int, default=6)
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--attack', choices=['all','random','rank1','adaptive'], default='all')
    parser.add_argument('--bench', action='store_true')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.model == 'mlp':
        model = MLP().to(device)
    else:
        model = Transformer(args.layers, nhead=args.heads).to(device)

    # build PNM
    pdict, masters, key = inject_pnm(model)
    # canary set
    canary_x = torch.randint(0, 1000, (64, 20)).to(device)
    with torch.no_grad():
        canary_y = model(canary_x)

    if args.bench:
        ok, t = bench(model, pdict, masters, canary_x, canary_y)
        print(f'full_verify {t*1000:.2f} ms, passed={ok}')
        return

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
