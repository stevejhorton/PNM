# PNM validation bundle – quick start

1. Install deps  
   pip install -q torch numpy tqdm

2. Run full battery (MLP + 6-layer transformer)  
   python validate.py --model transformer --layers 6 --heads 8 --attack all

3. See timings  
   python validate.py --bench
```
Tree:
pnm_validation_bundle/
├── README_QUICKSTART.md      # 3 commands to rerun everything
├── validate.py               # CLI entry point
├── attacks.py                # random, rank-1, adaptive, etc.
├── models.py                 # MLP + transformer builders
├── pnm_core.py               # parity node, master, HMAC helpers
├── notebooks/
│   └── reproduce_results.ipynb   # Jupyter version if preferred
└── checksums.sha256          # sha256 of clean weights used in bench
```
