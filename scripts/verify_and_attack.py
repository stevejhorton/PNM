import argparse, json, pathlib, torch, os, random
from pnm import SimpleMLP
from pnm.hardened_pnm import HPNMManager
from pnm.canaries import CanarySuite

def load_env(art=".artifacts"):
    model = SimpleMLP()
    model.load_state_dict(torch.load(pathlib.Path(art)/"model.pt", map_location="cpu"))
    mgr = HPNMManager(model)
    # reload placement
    place = json.loads((pathlib.Path(art)/"placement.json").read_text())
    mgr.pn_specs = []
    for spec in place["pn_specs"]:
        mgr.pn_specs.append(type("PS", (), spec))
    mgr.mn_specs = []
    for m in place["mn_specs"]:
        mgr.mn_specs.append(type("MS", (), m))
    mgr.fixed_core_master = place["fixed_core"]
    mgr.fixed_edge_master = place["fixed_edge"]
    keys = json.loads((pathlib.Path(art)/"keys.json").read_text())
    key_m = bytes.fromhex(keys["key_m"]); key_root = bytes.fromhex(keys["key_root"])
    return model, mgr, key_m, key_root

def attack_at_rest(model, m=20):
    # Modify random weights in-place (simulating file tamper before load)
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                W = mod.weight
                for _ in range(max(1, m//4)):
                    r = random.randrange(W.shape[0]); c = random.randrange(W.shape[1])
                    W[r,c] += 0.01

def attack_runtime(model, m=20):
    return attack_at_rest(model, m=m)

def attack_low_rank(model, scale=0.01):
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                W = mod.weight
                u = torch.randn(W.shape[0], 1)
                v = torch.randn(1, W.shape[1])
                W += scale * (u @ v)

def attack_random_small(model, m=100):
    return attack_at_rest(model, m=m)

def attack_unmonitored_bulk(model, m=50):
    # Try to avoid last layer by focusing on first layer only
    with torch.no_grad():
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear) and name.endswith("fc1"):
                W = mod.weight
                for _ in range(max(1, m//2)):
                    r = random.randrange(W.shape[0]); c = random.randrange(W.shape[1])
                    W[r,c] += 0.02

def verify_all(mgr, key_m, key_root, art=".artifacts"):
    fast_ok = mgr.fast_check(key_m, locked_map_path=str(pathlib.Path(art)/"hp_parity_map.json"))
    full_ok = mgr.full_check(key_m, key_root, locked_map_path=str(pathlib.Path(art)/"hp_parity_map.json"))
    return fast_ok, full_ok

def run_canaries(model, art=".artifacts"):
    can = CanarySuite.load(str(pathlib.Path(art)/"canaries.json"))
    ok, err = can.evaluate(model)
    return ok, err

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", type=str, default="none",
                    choices=["none","at_rest","runtime","low_rank","random_small","unmonitored_bulk"])
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--scale", type=float, default=0.01)
    ap.add_argument("--art", type=str, default=".artifacts")
    args = ap.parse_args()

    model, mgr, key_m, key_root = load_env(art=args.art)

    if args.attack != "none":
        print(f"Applying attack: {args.attack}")
        if args.attack == "at_rest":
            attack_at_rest(model, m=args.m)
        elif args.attack == "runtime":
            attack_runtime(model, m=args.m)
        elif args.attack == "low_rank":
            attack_low_rank(model, scale=args.scale)
        elif args.attack == "random_small":
            attack_random_small(model, m=args.m)
        elif args.attack == "unmonitored_bulk":
            attack_unmonitored_bulk(model, m=args.m)

    fast_ok, full_ok = verify_all(mgr, key_m, key_root, art=args.art)
    can_ok, err = run_canaries(model, art=args.art)

    print("=== Verification Results ===")
    print(f"Fast check (fixed masters): {fast_ok}")
    print(f"Full check (all masters + root): {full_ok}")
    print(f"Canaries: {can_ok} (max |err| = {err:.6f})")

if __name__ == "__main__":
    main()
