"""Track A: client simulator, local train/eval, timing, availability, FL loop.

Torch-dependent modules (`model`, `local`) are imported explicitly, never from here,
so torch-free sim modules stay importable in CI.
"""
