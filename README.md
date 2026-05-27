---
title: Resonance Fold
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: docker
app_file: apps/gradio/app.py
app_port: 7860
pinned: false
---

# 🧬 Resonance-Fold: Pure Math Protein Folding Engine

Welcome to the **Resonance-Fold** monorepo, a core repository of the **Nexus Resonance Codex (NRC)**. 

This repository houses a deterministic polymer physics engine for protein folding that operates with 100% mathematical precision on disordered proteins (IDPs) without stochastic runtime AI inference.

## 🏛 Project Architecture

This is a modern monorepo separating our high-performance mathematics from the user interface:

- `packages/nrc-physics-engine/`: The core physics engine implementing the Trageser Tensor Theorem (TTT-7), $\phi$-spiral mapping, and Lattice-Parity Embeddings (LPE).
- `apps/gradio/`: A premium, visual Gradio interface providing real-time 3D structure rendering, biophysical analytics (isoelectric point, hydropathy, charges), mutation simulations ($\Delta\Delta G$), and research package exports.

---
*Powered by Nexus Resonance Codex. Coding the Future of Biology.*
