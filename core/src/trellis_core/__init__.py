"""Engine-independent EXL3 pieces shared by every trellis-serve plugin.

- `trellis_core.format`: reads EXL3 checkpoints from safetensors headers (pure Python, no torch).
- `trellis_core.reference`: bit-exact PyTorch decoder for EXL3 linears; the ground truth every kernel is tested against.
"""
