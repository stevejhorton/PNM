import argparse, os, json, pathlib, torch, random
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from pnm import SimpleMLP
from pnm.hardened_pnm import HPNMManager
from pnm.canaries import CanarySuite

def make_data(n=100, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.linspace(-1, 1, n).view(-1,1)
    y = 2*x + 1 + 0.05*torch.randn(n,1, generator=gen)
    return x, y

def train(model, x, y, epochs=500, lr=1e-2, seed=0):
    torch.manual_seed(seed)
    ds = TensorDataset(x, y)
    dl = DataLoader(ds, batch_size=16, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = lossf(pred, yb)
            loss.backward()
            opt.step()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--num-parity", type=int, default=100)
    ap.add_argument("--num-masters", type=int, default=8)  # not used directly; fixed 2 for fast check + rest could be extended
    ap.add_argument("--out", type=str, default=".artifacts")
    args = ap.parse_args()

    pathlib.Path(args.out).mkdir(exist_ok=True, parents=True)

    # Keys (demo only; store securely in real deployments)
    key_m = os.urandom(32)
    key_root = os.urandom(32)
    (pathlib.Path(args.out)/"keys.json").write_text(json.dumps({
        "key_m": key_m.hex(),
        "key_root": key_root.hex()
    }, indent=2))

    x, y = make_data(seed=args.seed)
    model = SimpleMLP()
    train(model, x, y, epochs=500, lr=1e-2, seed=args.seed)
    torch.save(model.state_dict(), pathlib.Path(args.out)/"model.pt")

    # Place nodes and lock
    mgr = HPNMManager(model)
    mgr.place_nodes(num_parity=args.num_parity, seed=args.seed, density=args.density)
    locked, public_hash = mgr.lock(key_m, key_root, out_path=args.out)

    # Canaries
    can = CanarySuite.make_linear_suite(n=16)
    can.save(str(pathlib.Path(args.out)/"canaries.json"))

    print("=== Training + Lock Completed ===")
    print(f"Artifacts directory: {args.out}")
    print(f"Public hash: {public_hash}")
    print(f"Fixed masters: core={mgr.fixed_core_master}, edge={mgr.fixed_edge_master}")
    print(f"PN count: {len(mgr.pn_specs)}, MN count: {len(mgr.mn_specs)}")

if __name__ == "__main__":
    main()
