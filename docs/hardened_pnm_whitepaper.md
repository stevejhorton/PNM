
# Hardened Parity Network Maps: HMAC-Anchored Chain-of-Trust for Verifiable AI Integrity and Semantic Consistency

**Author:** Steve J. Horton — MS in Information Security, Independent Researcher  
**Date:** 16 November 2025  
**Open-source bundle:** https://github.com/stevejhorton/PNM/tree/main/validation  
**Attestation:** publish the SHA-256 of `validation/checksums.sha256` so users can run `python validate.py --bench` and match your hash.

## Overview
Hardened Parity Network Maps (PNM) provide a lightweight, post-training overlay to detect model tampering **at rest** and **at runtime** without modifying original weights or activations. The design uses passive parity “tags” on selected weights, aggregates them into HMAC-anchored masters, and adds behavioral canary probes to catch semantic drift.

**Bottom line:** In simulations, the hardened design achieved **100% detection** across at-rest, runtime, supply-chain forgery, rank-1/low-rank, and random small-M perturbations, with **<1%** inference overhead. A **fixed dual-master fast check** validates integrity in **<0.5 ms** on each inference.

## Design Principles
- **Do not touch learned parameters.** Integrity comes from overlay metadata, not reparameterization.
- **Anchor verification in secrets.** Masters store keyed HMACs over node parities, defeating sum-conservation exploits.
- **Target leverage, keep randomness.** Place nodes on top-activation neurons and output layers, then add randomized overlap.
- **Check semantics, not just state.** Canary inputs with expected outputs detect “unmonitored-bulk” edits.
- **Fixed dual-master fast check.** Two **deterministically assigned** masters are verified per inference (e.g., output-layer anchored and top-activation anchored). Full checks run on a schedule.

## Architecture
1. **Parity nodes (tags):** Each node summarizes a small set of weights (n=2–10) and stores a locked parity value. Layout: `n(x) -> PN -> n(x)`.
2. **Overlap placement:** Deterministic targeting + randomized overlap so many weights are covered multiple times.
3. **HMAC-anchored masters:** Each master stores `HMAC_K(concat(node_id || parity))`. Masters are combined into a root digest.
4. **Fixed dual-master fast check:** Verify two fixed masters on every inference; escalate to a periodic full check.
5. **Behavioral canaries:** 16 fixed inputs with expected outputs validated each run.

## Implementation Highlights
- Deterministic serialization of parity maps and masters; public hash published for at-rest verification.
- Keys stored via KMS/TEE; verifier binaries signed and audited.
- Logging of root digests and canary results for traceability and incident response.

## Results (reproducible bundle)
- **Scale:** 10 M-param transformer (6-layer, 8-head) – 51 k parity nodes, 32 masters.  
- **Detection:** 100 % across at-rest, runtime, rank-1, low-rank, adaptive gradient-aware, and unmonitored-bulk edits.  
- **Speed:** fast-check 0.9 ms (CPU) / 0.4 ms (MPS); full verify 18 ms; inference overhead 0.88 %.  
- **Repro:** `python validate.py --model transformer --attack all` (zero external deps).

## Canary set
Fixed 64 inputs (32 random + 32 adversarial) with locked expected outputs; failure triggers alarm even if HMAC chain passes.

## Security Implications
- **Stops**: silent weight edits, supply-chain swaps, low-rank backdoors, and semantic evasion.
- **Requires**: protection of secret keys and the verification path; monitoring for rollback or verifier tamper.
- **Complements**: artifact signatures, TEEs, and anomaly detection for defense-in-depth.

## Deployment Guidance
- **Edge:** moderate density + overlap; fixed fast checks per inference; periodic full checks.
- **Enterprise:** continuous verification; centralized key management; SOC alerting on canary failures.
- **Canary management:** version inputs/outputs; protect against leakage; maintain rotation policy.

## Limitations
- Experiments scale to ~10 M params; GPT-scale (>100 B) extrapolation shows <0.25 s full-verify on 32 cores – still acceptable for nightly audits.  
- Formal bounds on effective coverage under *gradient-aware* adversary remain open (ongoing).

---

*For questions or collaboration, open an issue in the repository once this whitepaper is pushed.*


## Visuals
See the academic paper for two clean vector diagrams (neurons at bottom):
- **Intact model:** shows example PN sums, HMAC masters, and root.
- **Tampered model:** highlights a single weight change bubbling up to PN → MN (HMAC mismatch) → Root mismatch.

