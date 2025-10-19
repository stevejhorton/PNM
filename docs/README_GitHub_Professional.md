# Hardened Parity Network Maps (PNM)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Hardened Parity Network Maps (PNM)** provide a verifiable, post-training mechanism for ensuring that AI models remain unmodified and trustworthy at rest and during runtime.  
PNM establishes a **cryptographically anchored chain of trust** across a model’s neurons, parity nodes, master nodes, and behavioral canaries—detecting even subtle tampering with negligible performance overhead.

---

## 🔍 Key Features
- **HMAC-Anchored Masters:** Prevent sum-conservation exploits through keyed cryptographic digests.  
- **Fixed Dual-Master Fast Check:** Deterministic core and edge masters verified each inference (sub-millisecond).  
- **Overlap Placement:** Focus coverage on top activations and outputs with randomized overlap for tamper resilience.  
- **Behavioral Canaries:** Detect semantic drift or unmonitored parameter edits.  
- **Chain-of-Trust Serialization:** Deterministic map hashing enables public verification of model integrity.  

---

## 🧩 Architecture
PNM layers sit *on top* of trained weights:
```
Neurons → Parity Nodes (PN) → Master Nodes (MN) → Root Digest → Verify()
```
A fixed dual-master fast check validates `M_core` and `M_edge` on every inference, while full verification recomputes all masters and the root HMAC periodically.

See the full diagrams in the `/docs` or in the academic paper for:  
- **Hierarchical Integrity Flow (Intact & Tampered)**  
- **Horizontal Verification Pipeline**

---

## 🚀 Quickstart
```bash
git clone https://github.com/stevejhorton/PNM.git
cd PNM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Train & lock demo model
python scripts/train_and_lock.py --seed 0 --density 0.5 --num-parity 100

# Verify integrity
python scripts/verify_and_attack.py --attack none

# Try simulated tampering
python scripts/verify_and_attack.py --attack at_rest --m 20
```
Artifacts are stored under `.artifacts/` and include the model, parity map, root digest, and canaries.

---

## ⚙️ How It Works
1. **Parity nodes** summarize small neuron groups via checksums.  
2. **Masters** compute HMACs over assigned parity values.  
3. **Root digest** chains master HMACs for full-model validation.  
4. **Fixed dual-master fast check** reads two masters each inference for sub-ms assurance.  
5. **Behavioral canaries** confirm semantic integrity even for unmonitored parameters.

---

## 📖 Citation
If you reference or extend this work, please cite:

```bibtex
@misc{horton2025pnm,
  title  = {Hardened Parity Network Maps: HMAC-Anchored Chain-of-Trust for Verifiable AI Integrity and Semantic Consistency},
  author = {Horton, Steve J.},
  year   = {2025},
  note   = {arXiv preprint arXiv:xxxx.xxxxx}
}
```

---

## 🧑‍💼 About the Author
**Steve J. Horton** is an information security professional with over three decades of experience in mission-critical communications and assurance.  
Beginning his career in the U.S. Air Force as a **3C2X1 Tech Controller**, he later spent **23 years at the Naval Research Laboratory’s Center for High Assurance Computing**, specializing in secure infrastructure and verification systems.  
He now serves as **VPN Service Level Owner for Optum/UHG**, a Fortune 4 enterprise.  
Steve holds an MS in Information Security and leads independent research on verifiable AI integrity and trust architectures.  

🔗 [LinkedIn](https://www.linkedin.com/in/steve-horton-8312199)

---

© 2025 Steve J. Horton — Released under the MIT License.
