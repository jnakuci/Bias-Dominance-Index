# Bias-Dominance-Index (BDI) Calculator

BDI Calculator is a lightweight PyTorch utility for computing the Bias Dominance Index (BDI) for individual model decisions. BDI decomposes a decision margin into input-dependent feature support and input-independent offset support, allowing users to assess whether a model’s decisiveness is primarily feature-supported or bias-driven.

The code supports aggregate and layer-resolved BDI estimates, automatic detection of supported bias-containing layers, and custom scalar margins for non-standard model outputs. It is intended for model auditing, interpretability research, and mechanism-aware analysis of neural-network confidence.
