import pathlib, json, torch, os
from pnm import SimpleMLP
from pnm.hardened_pnm import HPNMManager

def test_lock_and_verify(tmp_path):
    art = tmp_path / ".artifacts"
    art.mkdir(parents=True, exist_ok=True)
    key_m = os.urandom(32)
    key_root = os.urandom(32)

    # Model
    model = SimpleMLP()
    mgr = HPNMManager(model)
    mgr.place_nodes(num_parity=20, density=0.5, seed=0)
    locked, pub = mgr.lock(key_m, key_root, out_path=str(art))

    assert (art / "hp_parity_map.json").exists()
    assert len(locked.pn_values) > 0

    # Fast + Full verify on untampered model
    assert mgr.fast_check(key_m, locked_map_path=str(art / "hp_parity_map.json"))
    assert mgr.full_check(key_m, key_root, locked_map_path=str(art / "hp_parity_map.json"))
