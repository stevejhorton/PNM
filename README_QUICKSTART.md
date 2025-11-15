# PNM validation bundle – quick start

1. Install deps  
   pip install -q torch numpy tqdm

2. Run full battery (MLP + 6-layer transformer)  
   python validate.py --model transformer --layers 6 --heads 8 --attack all

3. See timings  
   python validate.py --bench
