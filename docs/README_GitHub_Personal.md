# Hardened Parity Network Maps (PNM)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Hardened Parity Network Maps (PNM)** is a hands-on framework for securing AI models after training—catching tampering at rest, in memory, or in the supply chain.  
It brings decades of network integrity and assurance principles into the AI era, creating a **living chain of trust** from neurons to master nodes.

---

## 🔍 Why It Exists
The more we rely on machine learning, the more dangerous quiet, hard-to-detect tampering becomes.  
PNM is built to make that impossible to hide — by giving models a self-verifiable heartbeat.

---

## 🧩 Core Design
- **HMAC-Anchored Verification:** cryptographically keyed masters seal every parity region.  
- **Fixed Dual-Master Fast Check:** constant-time root health check each inference.  
- **Parity Overlaps:** both strategic and random, catching subtle edits anywhere.  
- **Behavioral Canaries:** “known questions” the model must still answer correctly.  
- **Open, Reproducible Code:** transparent PyTorch implementation for validation.  

---

## 🚀 Quickstart
```bash
git clone https://github.com/stevejhorton/PNM.git
cd PNM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/train_and_lock.py --seed 0
python scripts/verify_and_attack.py --attack none
```

---

## 🧠 How It Works
Each parity node quietly mirrors a few trained weights. When any weight changes—even slightly—the parity sum and its master’s HMAC change, breaking the chain.  
The system checks two masters (core + edge) every inference in under a millisecond, while behavioral canaries keep watch over the model’s semantics.

---

## 📖 Citation
```bibtex
@misc{horton2025pnm,
  title  = {Hardened Parity Network Maps: HMAC-Anchored Chain-of-Trust for Verifiable AI Integrity and Semantic Consistency},
  author = {Horton, Steve J.},
  year   = {2025},
  note   = {arXiv preprint arXiv:xxxx.xxxxx}
}
```

---

## 👤 About the Author
**Steve J. Horton** has spent his life building, securing, and breaking complex systems to make them stronger.  
He started as a **Tech Controller (3C2X1)** in the **U.S. Air Force**, keeping global communications alive under pressure.  
For **23 years**, he worked at the **Naval Research Laboratory’s Center for High Assurance Computing**, helping design and test some of the most trusted infrastructure on the planet.  
Today, he’s the **VPN Service Level Owner for Optum/UHG**, applying the same mindset to one of the world’s largest enterprise networks — and exploring how those same assurance ideas can protect AI.  

🔗 [Connect on LinkedIn](https://www.linkedin.com/in/steve-horton-8312199)

---

© 2025 Steve J. Horton — MIT License.
