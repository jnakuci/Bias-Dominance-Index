# Bias-Dominance-Index (BDI) Calculator

BDI Calculator is a lightweight PyTorch utility for computing the Bias Dominance Index (BDI) for individual model decisions. BDI decomposes a decision margin into input-dependent feature support and input-independent offset support (i.e. bias term), allowing users to assess whether a model’s decisiveness is primarily feature-supported or bias-driven.

The implementation is configurable and can be adapted to specific architectures, layer definitions, margin functions, and bias-inclusion rules depending on the needs of a given analysis.

The current formulation supports additive offsets from `Linear`, `Conv1d`, `Conv2d`, `Conv3d`, `LayerNorm` and `Value-projection`  modules. 

For transformer models, query and key projection biases are excluded by default because they affect attention scores through interaction terms rather than contributing as direct additive evidence offsets. Value-projection biases are retained when implemented as standalone modules. Fused QKV projections are excluded conservatively by default unless architecture-specific slicing is implemented.

The package supports aggregate and layer-resolved BDI estimates, automatic detection of supported bias-containing layers, exclusion of query/key attention biases, and custom scalar margins for non-standard model outputs. It is intended for model auditing, interpretability research, and mechanism-aware analysis of neural-network confidence.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
