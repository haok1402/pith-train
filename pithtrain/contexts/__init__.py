"""
Process-global runtime state, one module per concern.

These values are established during setup and read frequently across the framework. Keeping them
here as module state lets code read them in-line instead of threading them through every
constructor and call. See each module for its exports.

Two stores are the exception to "set once at startup": weight_grad_store (the deferred weight
gradients) and fp8_weight_cache (the quantized-weight version counter) hold per-step state that
DualPipeV.step drives. Import them by class name.
"""
